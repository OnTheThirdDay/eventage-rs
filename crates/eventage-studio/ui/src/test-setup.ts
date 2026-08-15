/**
 * Browser APIs jsdom does not implement.
 *
 * The timeline measures itself with a ResizeObserver and paints on a canvas;
 * neither exists here. Stubbing them lets the component tree mount so the
 * surrounding behaviour — tabs, transport, selection, seeking — can be tested.
 * The drawing itself is verified through the pure geometry in `timeline.ts`.
 */

// The pure-logic suites run in the node environment, where none of this
// exists and none of it is needed.
if (typeof window !== "undefined") {

class StubResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= StubResizeObserver as unknown as typeof ResizeObserver;

const noop = () => {};
const stubContext = new Proxy(
  {
    canvas: null,
    setTransform: noop,
    clearRect: noop,
    fillRect: noop,
    beginPath: noop,
    moveTo: noop,
    lineTo: noop,
    arc: noop,
    arcTo: noop,
    closePath: noop,
    fill: noop,
    stroke: noop,
    fillText: noop,
    setLineDash: noop,
  },
  { get: (target, key) => (key in target ? Reflect.get(target, key) : noop), set: () => true },
);

HTMLCanvasElement.prototype.getContext = (() =>
  stubContext) as unknown as HTMLCanvasElement["getContext"];
}
