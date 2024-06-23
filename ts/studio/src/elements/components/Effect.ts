import { AudioComponent } from "../../abstracts/AudioComponent";
import { ControllerGroup, IEffect } from "../../schema";

export abstract class Effect extends AudioComponent implements IEffect {
    abstract plugin: string;
    controllerGroups: ControllerGroup[] = [];
}