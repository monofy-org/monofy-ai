var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
import EventObject from "../EventObject";
var BaseElement = /** @class */ (function (_super) {
    __extends(BaseElement, _super);
    function BaseElement(tagName, className) {
        var _this = _super.call(this) || this;
        _this.domElement = document.createElement(tagName);
        if (className) {
            _this.domElement.classList.add(className);
        }
        return _this;
    }
    BaseElement.prototype.dispose = function () {
        this.removeAllListeners();
        this.domElement.remove();
    };
    BaseElement.prototype.appendChild = function (child) {
        this.domElement.appendChild(child.domElement);
        return this;
    };
    BaseElement.prototype.removeChild = function (child) {
        this.domElement.removeChild(child.domElement);
        return this;
    };
    return BaseElement;
}(EventObject));
export { BaseElement };
