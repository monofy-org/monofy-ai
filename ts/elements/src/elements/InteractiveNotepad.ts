export class InteractiveNotepad {
    domElement: HTMLDivElement;
    textArea: HTMLTextAreaElement;
    constructor() {
        this.domElement = document.createElement('div');
        this.textArea = document.createElement('textarea');
        this.textArea.style.width = '100%';
        this.textArea.style.height = '100%';
        this.textArea.style.border = 'none';
        this.textArea.style.resize = 'none';
        this.textArea.style.outline = 'none';
        this.textArea.style.padding = '10px';
        this.textArea.style.fontSize = '16px';
        this.textArea.style.fontFamily = 'Arial';
        this.textArea.style.color = '#333';
        this.textArea.style.backgroundColor = '#fff';
        this.textArea.style.overflow = 'auto';
        this.domElement.appendChild(this.textArea);        
    }
}
