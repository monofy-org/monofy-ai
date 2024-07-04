import { AudioComponent } from "../../abstracts/AudioComponent";
import { EffectWindow } from "../../abstracts/EffectWindow";
import { IEffect } from "../../schema";

export abstract class Effect extends AudioComponent implements IEffect {
    abstract plugin: string;
    abstract window?: EffectWindow;    
}