/**
 * claw-whatsapp-bridge
 *
 * Bridges WhatsApp (via Baileys) ↔ eventage-claw (via HTTP channel).
 *
 * Flow:
 *   WhatsApp message → POST /message on claw's HTTP channel (port 3000)
 *   claw agent responds → ChannelOutputWorker POSTs to /send on this bridge (port 3001)
 *   /send handler → sock.sendMessage(jid, { text }) back to WhatsApp
 *
 * Configuration (env vars):
 *   CLAW_HTTP_URL   — URL of claw's HTTP channel  (default: http://localhost:3000)
 *   CLAW_GROUP      — default group to deliver messages to  (default: personal)
 *   BRIDGE_PORT     — port this bridge listens on for /send  (default: 3001)
 *   TRIGGER_PREFIX  — only forward messages starting with this (default: none = all messages)
 *   AUTH_DIR        — where to store Baileys auth state  (default: ./auth)
 *   PAIRING_CODE    — phone number for pairing-code auth instead of QR  (e.g. +1234567890)
 *
 * Usage:
 *   npm install
 *   node bridge.js
 */

import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  downloadMediaMessage,
} from '@whiskeysockets/baileys';
import express from 'express';
import qrcode from 'qrcode-terminal';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';
import pino from 'pino';

// ── Config ────────────────────────────────────────────────────────────────────

const CLAW_HTTP_URL    = process.env.CLAW_HTTP_URL    ?? 'http://localhost:3000';
const CLAW_GROUP       = process.env.CLAW_GROUP       ?? 'personal';
const BRIDGE_PORT      = parseInt(process.env.BRIDGE_PORT ?? '3001', 10);
const TRIGGER_PREFIX   = process.env.TRIGGER_PREFIX   ?? '';    // '' = respond to everything
const AUTH_DIR         = process.env.AUTH_DIR ?? path.join(path.dirname(fileURLToPath(import.meta.url)), 'auth');
const PAIRING_NUMBER   = process.env.PAIRING_CODE ?? '';        // e.g. '+1234567890'
// Set SELF_CHAT=true to respond when you message your own "Saved Messages" / Note to Self.
const SELF_CHAT        = process.env.SELF_CHAT === 'true';
// Directory where downloaded media files are saved for the agent to process.
// Should be inside the agent's work_dir (default: ~/.claw/groups/<group>/attachments).
const ATTACHMENTS_DIR  = process.env.ATTACHMENTS_DIR
  ?? path.join(process.env.HOME ?? '.', '.claw', 'groups', CLAW_GROUP, 'attachments');

// Optional sender→group routing map.
// SENDER_MAP="61400000000:alice,61411111111:bob"
// Phone numbers not in the map fall through to CLAW_GROUP (your default group).
const SENDER_MAP = new Map(
  (process.env.SENDER_MAP ?? '').split(',').flatMap(entry => {
    const [phone, group] = entry.split(':').map(s => s.trim());
    return phone && group ? [[phone, group]] : [];
  })
);

/** Resolve which claw group a message should be routed to. */
function resolveGroup(jid) {
  const phone = jid.split('@')[0].split(':')[0];
  return SENDER_MAP.get(phone) ?? CLAW_GROUP;
}


// Suppress Baileys internal logs (very noisy). Set BAILEYS_LOG=debug to see them.
const logger = pino({ level: process.env.BAILEYS_LOG ?? 'silent' });

// ── Bridge state ──────────────────────────────────────────────────────────────

/** Active Baileys socket — set after connection. */
let sock = null;

/**
 * True once connection.update fires with connection='open'.
 * Messages arriving before this point are queued and replayed afterwards.
 * Needed because Baileys can fire messages.upsert before the socket is fully
 * authenticated: sock.user.id is not yet available, so selfJid resolution fails
 * (SELF_CHAT mode) and session keys may not be ready for decryption.
 */
let connected = false;
const preConnectQueue = [];

/**
 * IDs of messages the bridge itself sent.
 * WhatsApp reflects outgoing messages back to all linked devices as fromMe:true
 * notifications. We track sent IDs so we can skip these echoes and avoid
 * forwarding our own responses back to claw (which would cause a reply loop).
 * Entries are removed after 60 s to prevent unbounded growth.
 */
const sentMessageIds = new Set();

/**
 * IDs of inbound messages already forwarded to claw.
 * WhatsApp linked devices can receive the same incoming message twice: once
 * via direct push (type='notify') and once via device-to-device message sync
 * (also type='notify'). Without deduplication the bridge forwards both copies,
 * causing claw to process the same request twice and produce duplicate replies.
 * Entries expire after 60 s.
 */
const seenInboundIds = new Set();

// ── Express server (receives claw → WhatsApp responses) ───────────────────────

const app = express();
app.use(express.json());

/**
 * POST /send
 * Body: { reply_to: "<jid>", text: "<response text>", group?: "personal" }
 *
 * Called by eventage-claw's ChannelOutputWorker when the agent finishes a turn.
 */
app.post('/send', async (req, res) => {
  const { reply_to: jid, text } = req.body;

  if (!jid || !text) {
    return res.status(400).json({ ok: false, error: 'reply_to and text are required' });
  }

  if (!sock) {
    return res.status(503).json({ ok: false, error: 'WhatsApp not connected' });
  }

  try {
    // Split long responses (WhatsApp has a ~65535 char limit per message).
    const chunks = splitMessage(text, 4000);
    for (const chunk of chunks) {
      const sent = await sock.sendMessage(jid, { text: chunk });
      // Track the sent message ID so the echo (reflected back to this linked
      // device as fromMe:true) is not re-forwarded to claw.
      if (sent?.key?.id) {
        sentMessageIds.add(sent.key.id);
        setTimeout(() => sentMessageIds.delete(sent.key.id), 60_000);
      }
    }
    console.log(`[bridge] → WhatsApp ${jid.split('@')[0]}: ${text.slice(0, 60)}…`);
    res.json({ ok: true });
  } catch (err) {
    console.error('[bridge] sendMessage error:', err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

/** GET /health — liveness check */
app.get('/health', (_req, res) => {
  res.json({ ok: true, connected: sock !== null });
});

app.listen(BRIDGE_PORT, () => {
  console.log(`[bridge] Outbound server listening on port ${BRIDGE_PORT}`);
});

// ── WhatsApp connection ────────────────────────────────────────────────────────

async function connectToWhatsApp() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  sock = makeWASocket({
    version,
    logger,
    auth: state,
    printQRInTerminal: false,   // we handle QR ourselves
    browser: ['claw-bridge', 'Chrome', '120.0'],
    getMessage: async () => undefined, // don't load message history
  });

  // ── Auth events ────────────────────────────────────────────────────────────

  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      if (PAIRING_NUMBER) {
        // Use pairing code instead of QR (better for headless environments).
        const code = await sock.requestPairingCode(PAIRING_NUMBER.replace(/\D/g, ''));
        console.log(`\n[bridge] Pairing code for ${PAIRING_NUMBER}: ${code}\n`);
        console.log('[bridge] Enter this code in WhatsApp → Linked Devices → Link a Device → Link with phone number\n');
      } else {
        console.log('\n[bridge] Scan the QR code below with WhatsApp → Linked Devices → Link a Device\n');
        qrcode.generate(qr, { small: true });
      }
    }

    if (connection === 'close') {
      connected = false;
      const reason = lastDisconnect?.error?.output?.statusCode;
      const shouldReconnect = reason !== DisconnectReason.loggedOut;
      console.log('[bridge] Connection closed. Reason:', reason, shouldReconnect ? '— reconnecting…' : '— logged out.');
      if (shouldReconnect) {
        setTimeout(connectToWhatsApp, 3000);
      } else {
        console.log('[bridge] Logged out. Delete the auth/ folder and restart to re-authenticate.');
        process.exit(1);
      }
    }

    if (connection === 'open') {
      connected = true;
      const jid = sock.user?.id ?? '?';
      console.log(`[bridge] ✓ Connected to WhatsApp as ${jid.split(':')[0]}`);
      console.log(`[bridge] Forwarding to claw group "${CLAW_GROUP}" at ${CLAW_HTTP_URL}`);
      if (TRIGGER_PREFIX) {
        console.log(`[bridge] Trigger prefix: "${TRIGGER_PREFIX}"`);
      } else {
        console.log('[bridge] No trigger prefix — responding to all direct messages');
      }

      // Flush any messages that arrived before the connection was fully open.
      // sock.user.id is now available so selfJid resolves correctly.
      if (preConnectQueue.length > 0) {
        console.log(`[bridge] Replaying ${preConnectQueue.length} queued message(s) received before connection was ready`);
        const queued = preConnectQueue.splice(0);
        for (const msg of queued) {
          await handleMessage(msg);
        }
      }
    }
  });

  sock.ev.on('creds.update', saveCreds);

  // ── Inbound messages ───────────────────────────────────────────────────────

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    // Accept both 'notify' (real-time) and 'append' (offline-sync on reconnect).
    // Pending messages sent while the bridge was offline can arrive as 'append'.
    if (type !== 'notify' && type !== 'append') return;

    console.log(`[bridge] upsert type=${type} count=${messages.length}`);

    for (const msg of messages) {
      if (!connected) {
        console.log(`[bridge] queuing id=${msg.key?.id} (not yet connected)`);
        preConnectQueue.push(msg);
        continue;
      }
      await handleMessage(msg);
    }
  });
}

// ── Per-message handler ────────────────────────────────────────────────────────

async function handleMessage(msg) {
  const msgId = msg.key?.id ?? '?';
  const msgTypes = Object.keys(msg.message ?? {}).join(',') || 'null';
  console.log(`[bridge] handle id=${msgId} fromMe=${msg.key?.fromMe} jid=${msg.key?.remoteJid} types=${msgTypes}`);

  // Skip status broadcasts always.
  if (msg.key.remoteJid === 'status@broadcast') {
    console.log(`[bridge] skip: status broadcast`);
    return;
  }

  // Skip messages the bridge itself sent (echoed back by WhatsApp to all
  // linked devices). Without this, claw's responses would be re-forwarded
  // to claw creating an infinite reply loop.
  if (msg.key.id && sentMessageIds.has(msg.key.id)) {
    console.log(`[bridge] skip: own echo id=${msg.key.id}`);
    return;
  }

  // Deduplicate: skip if we already successfully forwarded this message ID.
  // NOTE: we only mark an ID as seen AFTER extracting text below.  WhatsApp
  // delivers the same message ID multiple times: first as a CIPHERTEXT stub
  // (stub=2, types=null) while the session is being established, then again
  // with the actual decrypted content.  Marking seen on the first (empty)
  // arrival would deduplicate the second (content-bearing) arrival.
  if (msg.key.id && seenInboundIds.has(msg.key.id)) {
    console.log(`[bridge] skip: duplicate id=${msg.key.id}`);
    return;
  }

  // In SELF_CHAT mode: process only messages YOU sent (to yourself / Saved Messages).
  // In normal mode: process only messages from others.
  if (SELF_CHAT) {
    if (!msg.key.fromMe) {
      console.log(`[bridge] skip: not fromMe (SELF_CHAT mode)`);
      return;
    }
  } else {
    if (msg.key.fromMe) {
      console.log(`[bridge] skip: fromMe (normal mode)`);
      return;
    }
  }

  const rawJid = msg.key.remoteJid;
  const text = extractText(msg);
  const attachmentMeta = detectAttachment(msg);

  if (!text && !attachmentMeta) {
    // No content yet (e.g. CIPHERTEXT stub while session is being established).
    // Do NOT mark this ID as seen — the same ID will arrive again once Baileys
    // completes decryption, and that delivery will carry the actual content.
    console.log(`[bridge] skip: no text/attachment (types=${msgTypes} stub=${msg.messageStubType ?? 'none'})`);
    return;
  }

  // We have content — mark the ID as seen now to deduplicate any further
  // re-deliveries of the same message.
  if (msg.key.id) {
    seenInboundIds.add(msg.key.id);
    setTimeout(() => seenInboundIds.delete(msg.key.id), 60_000);
  }

  // Download any media attachment so the agent can process it via RunCommandTool.
  let attachments = [];
  if (attachmentMeta) {
    const att = await downloadAttachment(msg, attachmentMeta);
    if (att) attachments.push(att);
  }

  // Use a type-appropriate placeholder when there is no caption text.
  const placeholderText = attachmentMeta?.type === 'audio'    ? '[Voice message]'
                        : attachmentMeta?.type === 'image'    ? '[Image]'
                        : attachmentMeta?.type === 'video'    ? '[Video]'
                        : attachmentMeta?.type === 'document' ? `[Document: ${attachmentMeta.filename ?? ''}]`
                        : '';
  const messageText = text ?? placeholderText;

  // Apply trigger prefix filter (only to text portion).
  if (TRIGGER_PREFIX && !messageText.startsWith(TRIGGER_PREFIX)) {
    console.log(`[bridge] skip: prefix mismatch`);
    return;
  }

  const body = TRIGGER_PREFIX
    ? messageText.slice(TRIGGER_PREFIX.length).trim()
    : messageText;

  if (!body && attachments.length === 0) return;

  // In SELF_CHAT mode (Note to Self), LIDs are unreliable for sending back.
  // Reply to our own @s.whatsapp.net JID so the message appears in Note to Self.
  // In normal mode (incoming from others), use their JID directly.
  const selfJid = sock?.user?.id
    ? sock.user.id.split(':')[0] + '@s.whatsapp.net'
    : rawJid;
  const replyJid = SELF_CHAT ? selfJid : rawJid;

  const senderPhone = rawJid.split('@')[0];
  console.log(`[bridge] ← WhatsApp ${senderPhone}: ${body.slice(0, 80)}`);
  console.log(`[bridge] reply_to=${replyJid}`);

  // Forward to claw's HTTP channel.
  try {
    const res = await fetch(`${CLAW_HTTP_URL}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        group: resolveGroup(rawJid),
        text: body,
        sender: rawJid,
        reply_to: replyJid,   // ChannelOutputWorker uses this to route the response
        ...(attachments.length > 0 ? { attachments } : {}),
      }),
    });

    if (res.ok) {
      console.log(`[bridge] ✓ forwarded to claw`);
    } else {
      const err = await res.text();
      console.error(`[bridge] claw rejected message: ${res.status} ${err}`);
    }
  } catch (err) {
    console.error('[bridge] Failed to forward to claw:', err.message);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Extract plain text from a Baileys message object. */
function extractText(msg) {
  const m = msg.message;
  if (!m) return null;
  return (
    m.conversation ||
    m.extendedTextMessage?.text ||
    m.imageMessage?.caption ||
    m.videoMessage?.caption ||
    m.buttonsResponseMessage?.selectedDisplayText ||
    m.listResponseMessage?.title ||
    null
  );
}

/**
 * Detect a media attachment in the message and return metadata.
 * Returns null if the message carries no downloadable media.
 */
function detectAttachment(msg) {
  const m = msg.message;
  if (!m) return null;
  if (m.audioMessage)    return { type: 'audio',    mime: m.audioMessage.mimetype    ?? 'audio/ogg',         ext: 'ogg'  };
  if (m.pttMessage)      return { type: 'audio',    mime: m.pttMessage.mimetype      ?? 'audio/ogg',         ext: 'ogg'  };
  if (m.imageMessage)    return { type: 'image',    mime: m.imageMessage.mimetype    ?? 'image/jpeg',        ext: 'jpg'  };
  if (m.videoMessage)    return { type: 'video',    mime: m.videoMessage.mimetype    ?? 'video/mp4',         ext: 'mp4'  };
  if (m.documentMessage) return { type: 'document', mime: m.documentMessage.mimetype ?? 'application/octet-stream',
                                  ext: m.documentMessage.fileName?.split('.').pop() ?? 'bin',
                                  filename: m.documentMessage.fileName ?? '' };
  return null;
}

/**
 * Download media for a message and save it to ATTACHMENTS_DIR.
 * Returns an attachment descriptor object, or null on failure.
 */
async function downloadAttachment(msg, meta) {
  try {
    await fs.promises.mkdir(ATTACHMENTS_DIR, { recursive: true });
    const buffer = await downloadMediaMessage(msg, 'buffer', {});
    const filename = `${msg.key.id}.${meta.ext}`;
    const filepath = path.join(ATTACHMENTS_DIR, filename);
    await fs.promises.writeFile(filepath, buffer);
    console.log(`[bridge] downloaded ${meta.type} → ${filepath}`);
    return { type: meta.type, mime: meta.mime, path: filepath, filename: meta.filename ?? filename };
  } catch (err) {
    console.error(`[bridge] media download failed: ${err.message}`);
    return null;
  }
}

/** Split a long text into chunks without cutting words. */
function splitMessage(text, maxLen) {
  if (text.length <= maxLen) return [text];
  const chunks = [];
  let pos = 0;
  while (pos < text.length) {
    let end = pos + maxLen;
    if (end < text.length) {
      // Try to break at last newline or space.
      const nl = text.lastIndexOf('\n', end);
      const sp = text.lastIndexOf(' ', end);
      const cut = Math.max(nl, sp);
      if (cut > pos) end = cut + 1;
    }
    chunks.push(text.slice(pos, end).trim());
    pos = end;
  }
  return chunks.filter(Boolean);
}

// ── Entry point ───────────────────────────────────────────────────────────────

/**
 * Delete stale Signal session files from the auth directory on startup.
 *
 * When the bridge restarts, the in-memory Signal sessions are gone but
 * WhatsApp's server still holds the old session keys. This mismatch causes
 * "Bad MAC / Session error" on the first message while Baileys renegotiates a
 * new prekey bundle — during which extractText() returns null and the message
 * is silently dropped.
 *
 * Deleting session-*.json forces Baileys to establish a fresh session on first
 * connect without triggering Bad MAC errors. We keep creds.json so the device
 * registration / pairing is preserved (no QR re-scan needed).
 */
function clearStaleSignalSessions(authDir) {
  let deleted = 0;
  try {
    const files = fs.readdirSync(authDir);
    for (const f of files) {
      if (f.startsWith('session-') && f.endsWith('.json')) {
        fs.unlinkSync(path.join(authDir, f));
        deleted++;
      }
    }
  } catch {
    // Auth dir doesn't exist yet — first run, nothing to clear.
  }
  if (deleted > 0) {
    console.log(`[bridge] Cleared ${deleted} stale Signal session file(s) from ${authDir}`);
  }
}

console.log('[bridge] Starting claw WhatsApp bridge…');
clearStaleSignalSessions(AUTH_DIR);
connectToWhatsApp().catch((err) => {
  console.error('[bridge] Fatal:', err);
  process.exit(1);
});
