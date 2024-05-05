export interface EventDataMap {
  start: any;
  stop: any;
  pause: any;
  render: any;
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

  fireEvent(eventName: E, eventData?: EventDataMap[E]) {
    const onceCallbacks = this._onceEvents[eventName];
    if (onceCallbacks) {
      for (const callback of onceCallbacks) {
        try {
          callback(eventData);
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
          callback(eventData);
        } catch (error) {
          console.error(
            `Error executing .on callback for event: ${eventName}`,
            error
          );
        }
      }
    }
  }
}
