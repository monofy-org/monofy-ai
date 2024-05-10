import { EventDataMap } from "../../../../elements/src/EventObject";
import { BaseElement } from "../../../../elements/src/elements/BaseElement";

export class SharedComponents {
  static _components: { [name: string]: BaseElement<keyof EventDataMap> } = {};

  static getComponent<T extends BaseElement<keyof EventDataMap>>(name: string): T {
    return this._components[name] as T;
  }

  static add(name: string, component: BaseElement<keyof EventDataMap>) {
    this._components[name] = component;
  }  
}
