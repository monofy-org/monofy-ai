import { BaseElement } from "./elements/BaseElement";

export interface IBaseEvent {
  target: BaseElement;
  event: PointerEvent;
}

export interface IDragEvent extends IBaseEvent {
  top: number;
  left: number;
  deltaX: number;
  deltaY: number;
}

export interface IResizeEvent extends IBaseEvent {
  width: number;
  height: number;
}

export interface EventDataMap {
  cancel: unknown;
  start: unknown;
  stop: unknown;
  pause: unknown;
  result: unknown;
  update: unknown;
  open: unknown;
  close: unknown;
  edit: unknown;
  release: unknown;
  resize: IResizeEvent;
  select: IBaseEvent;
  scroll: WheelEvent;
  add: unknown;
  remove: unknown;
  change: unknown;
  drag: IDragEvent;
}

export type BaseEvent = keyof EventDataMap;

export default abstract class EventObject<E extends BaseEvent = BaseEvent> {
  private _events: {
    [eventName in E]?: ((e: EventDataMap[eventName]) => void)[];
  } = {};
  private _onceEvents: {
    [eventName in E]?: ((e: EventDataMap[eventName]) => void)[];
  } = {};

  on(eventName: E, callback: (e: EventDataMap[E]) => void) {
    if (!this._events[eventName]) {
      this._events[eventName] = [];
    }
    this._events[eventName]!.push(callback);

    return this;
  }

  once(eventName: E, callback: (e: EventDataMap[E]) => void) {
    if (!this._onceEvents[eventName]) {
      this._onceEvents[eventName] = [];
    }
    this._onceEvents[eventName]!.push(callback);

    return this;
  }

  emit(eventName: E, eventData?: unknown) {
    const onceCallbacks = this._onceEvents[eventName];
    if (onceCallbacks) {
      for (const callback of onceCallbacks) {
        try {
          callback(eventData as EventDataMap[E]);
        } catch (error) {
          console.error(
            `Error executing .once callback for event: ${eventName}`,
            error
          );
        }
      }
      this._onceEvents[eventName] = undefined;
    }

    const callbacks = this._events[eventName];
    if (callbacks) {
      for (const callback of callbacks) {
        try {
          callback(eventData as EventDataMap[E]);
        } catch (error) {
          console.error(
            `Error executing .on callback for event: ${eventName}`,
            error
          );
        }
      }
    }
  }

  removeAllListeners() {
    this._events = {};
    this._onceEvents = {};
  }
}
