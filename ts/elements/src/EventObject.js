var EventObject = /** @class */ (function () {
    function EventObject() {
        this._events = {};
        this._onceEvents = {};
    }
    EventObject.prototype.on = function (eventName, callback) {
        if (!this._events[eventName]) {
            this._events[eventName] = [];
        }
        this._events[eventName].push(callback);
        return this;
    };
    EventObject.prototype.once = function (eventName, callback) {
        if (!this._onceEvents[eventName]) {
            this._onceEvents[eventName] = [];
        }
        this._onceEvents[eventName].push(callback);
        return this;
    };
    EventObject.prototype.emit = function (eventName, eventData) {
        var onceCallbacks = this._onceEvents[eventName];
        if (onceCallbacks) {
            for (var _i = 0, onceCallbacks_1 = onceCallbacks; _i < onceCallbacks_1.length; _i++) {
                var callback = onceCallbacks_1[_i];
                try {
                    callback(eventData);
                }
                catch (error) {
                    console.error("Error executing .once callback for event: ".concat(eventName), error);
                }
            }
            this._onceEvents[eventName] = undefined;
        }
        var callbacks = this._events[eventName];
        if (callbacks) {
            for (var _a = 0, callbacks_1 = callbacks; _a < callbacks_1.length; _a++) {
                var callback = callbacks_1[_a];
                try {
                    callback(eventData);
                }
                catch (error) {
                    console.error("Error executing .on callback for event: ".concat(eventName), error);
                }
            }
        }
    };
    EventObject.prototype.removeAllListeners = function () {
        this._events = {};
        this._onceEvents = {};
    };
    return EventObject;
}());
export default EventObject;
