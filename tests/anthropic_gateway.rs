//! Routing the Messages API through a gateway.
//!
//! Portkey, LiteLLM, Helicone and the various Bedrock proxies all present the
//! same Anthropic API at a different address and route on their own headers,
//! so none of them needs a provider of its own — only an endpoint, a bearer
//! credential and somewhere to put headers. This checks the request that
//! actually leaves the process, against a server that records it, because
//! the failure mode otherwise is a 401 from a third party.

use eventage::llm::anthropic::AnthropicProvider;
use eventage::llm::types::ChatMessage;
use eventage::llm::LlmProvider;
use std::sync::{Arc, Mutex};

/// A stub Messages endpoint that records what it was sent.
struct Recorder {
    headers: Arc<Mutex<Vec<(String, String)>>>,
    path: Arc<Mutex<String>>,
}

impl Recorder {
    async fn start() -> (String, Self) {
        let headers: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let path = Arc::new(Mutex::new(String::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (h, p) = (Arc::clone(&headers), Arc::clone(&path));
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();

            let mut lines = request.lines();
            if let Some(start) = lines.next() {
                *p.lock().unwrap() = start.split_whitespace().nth(1).unwrap_or("").to_string();
            }
            for line in lines {
                if line.is_empty() {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    h.lock()
                        .unwrap()
                        .push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
                }
            }

            let body = r#"{"id":"m","type":"message","role":"assistant","model":"m",
                "content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
                "usage":{"input_tokens":1,"output_tokens":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
        });

        (format!("http://{addr}"), Self { headers, path })
    }

    fn header(&self, name: &str) -> Option<String> {
        let headers = self.headers.lock().unwrap();
        headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
    }
}

#[tokio::test]
async fn a_gateway_gets_the_endpoint_the_bearer_token_and_its_routing_headers() {
    let (base, recorder) = Recorder::start().await;

    let provider = AnthropicProvider::new("pk-secret", "claude-sonnet-4-5")
        .with_base_url(&base)
        .with_bearer_auth(true)
        .with_headers([
            ("x-portkey-api-key", "pk-secret"),
            ("x-portkey-provider", "@my-provider"),
        ]);

    provider
        .complete(vec![ChatMessage::user("hi")], vec![])
        .await
        .expect("the gateway should be reachable");

    // The path is the standard one appended to whatever base it was given —
    // Portkey documents a base with no `/v1`, which lands here correctly.
    assert_eq!(*recorder.path.lock().unwrap(), "/v1/messages");

    // A gateway takes a bearer token, not `x-api-key`, and must not be sent
    // both: the second would be a credential leaked to a third party.
    assert_eq!(
        recorder.header("authorization").as_deref(),
        Some("Bearer pk-secret")
    );
    assert_eq!(recorder.header("x-api-key"), None);

    assert_eq!(
        recorder.header("x-portkey-provider").as_deref(),
        Some("@my-provider")
    );
    // And the API version still goes, because it is still that API.
    assert!(recorder.header("anthropic-version").is_some());
}

#[tokio::test]
async fn talking_to_anthropic_directly_still_uses_the_api_key_header() {
    // The other half of the switch: adding gateway support must not change
    // how the provider authenticates against Anthropic itself.
    let (base, recorder) = Recorder::start().await;

    AnthropicProvider::new("sk-ant-real", "claude-sonnet-4-5")
        .with_base_url(&base)
        .complete(vec![ChatMessage::user("hi")], vec![])
        .await
        .unwrap();

    assert_eq!(recorder.header("x-api-key").as_deref(), Some("sk-ant-real"));
    assert_eq!(recorder.header("authorization"), None);
}
