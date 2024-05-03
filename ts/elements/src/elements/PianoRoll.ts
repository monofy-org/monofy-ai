const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const keyColors = ['white', 'black', 'white', 'black', 'white', 'white', 'black', 'white', 'black', 'white', 'black', 'white'];

export class Composition {
    title: string;
    description: string;
    tempo: number;
    events: GridItem[] = [];

    constructor() {
        this.title = 'Untitled';
        this.description = 'No description';
        this.tempo = 120;
    }
}

export interface IGridItem {
    pitch: number;
    start: number;
    end: number;
    label: string;
}

export class GridItem implements IGridItem {
    pitch: number;
    start: number;
    end: number;
    label: string | "" = "";
    domElement: HTMLDivElement;

    constructor(private readonly grid: Grid, item: IGridItem) {
        this.pitch = item.pitch;
        this.start = item.start;
        this.end = item.end;
        this.domElement = document.createElement('div');
        this.domElement.style.position = 'absolute';
        this.domElement.style.backgroundColor = 'blue';
        this.domElement.style.border = '1px solid #000';
        this.domElement.style.borderRadius = '5px';
        this.domElement.style.boxSizing = 'border-box';
        this.domElement.style.overflow = 'hidden';
        this.domElement.style.textOverflow = 'ellipsis';
        this.domElement.style.whiteSpace = 'nowrap';

        if (item.label) this.domElement.textContent = item.label;
        this.update()

        grid.domElement.appendChild(this.domElement);
    }

    update() {
        this.domElement.style.top = `${this.pitch * this.grid.noteHeight}px`;
        this.domElement.style.left = `${this.start * this.grid.beatWidth}%`;
        this.domElement.style.width = `${(this.end - this.start) * 100}%`;
    }
}

export class Grid {
    domElement: HTMLDivElement;
    gridElement: HTMLDivElement;
    notes: GridItem[] = [];
    currentNote: GridItem | null = null;
    noteHeight = 10;
    beatWidth = 100;

    constructor() {
        this.domElement = document.createElement('div');
        this.gridElement = document.createElement('div');
        this.domElement.appendChild(this.gridElement);

        this.gridElement.style.position = 'relative';
        this.gridElement.style.width = '100%';
        this.gridElement.style.height = '100%';
        this.gridElement.style.overflow = 'hidden';

        this.gridElement.addEventListener('pointerdown', (event) => {
            if (event.target === this.gridElement) {

                const pitch = Math.floor(event.layerY / this.noteHeight);

                this.currentNote = new GridItem(this, {
                    pitch: pitch,
                    start: event.clientX / this.gridElement.clientWidth,
                    end: event.clientX / this.gridElement.clientWidth,
                    label: getNoteNameFromPitch(pitch)
                });
                this.add(this.currentNote);
            }
        });

        function getNoteNameFromPitch(pitch: number): string {            
            const note = notes[pitch % notes.length];
            return note + Math.floor(pitch / notes.length);
        }

        this.gridElement.addEventListener('pointermove', (event) => {
            if (this.currentNote) {
                this.currentNote.end = event.clientX / this.gridElement.clientWidth;
                this.currentNote.domElement.style.width = `${(this.currentNote.end - this.currentNote.start) * 100}%`;
            }
        });

        this.gridElement.addEventListener('pointerup', (event) => {
            this.currentNote = null;
        });

        this.gridElement.addEventListener('pointerleave', (event) => {
            if (this.currentNote) {
                this.remove(this.currentNote);
                this.currentNote = null;
            }
        });

        this.gridElement.addEventListener('contextmenu', (event) => {
            event.preventDefault();            
        });

        this.gridElement.addEventListener('wheel', (event) => {
            if (this.currentNote) {
                this.currentNote.pitch += event.deltaY / 100;
                this.currentNote.domElement.style.top = `${this.currentNote.pitch * 10}px`;
            }
        });

        this.gridElement.addEventListener('pointerdown', (event) => {
            if (event.target instanceof HTMLDivElement) {
                const note = this.notes.find(n => n.domElement === event.target);
                if (note) {
                    this.currentNote = note;

                    if (event.button === 2) {
                        this.remove(note);
                        this.currentNote = null;
                    }
                }
            }
        });

        this.gridElement.addEventListener('pointermove', (event) => {
            if (this.currentNote) {
                this.currentNote.start = event.clientX / this.gridElement.clientWidth;
                this.currentNote.end = this.currentNote.start + parseFloat(this.currentNote.domElement.style.width) / this.gridElement.clientWidth;
                this.currentNote.domElement.style.left = `${this.currentNote.start * 100}%`;
            }
        });

        this.gridElement.addEventListener('pointerup', (event) => {
            this.currentNote = null;
        });

        this.gridElement.addEventListener('pointerleave', (event) => {
            this.currentNote = null;
        });

    }
     
    add(note: GridItem) {
        this.notes.push(note);
        this.gridElement.appendChild(note.domElement);
    }
    remove(note: GridItem) {
        this.notes = this.notes.filter(n => n !== note);
        this.gridElement.removeChild(note.domElement);
    }
    load(composition: object) {
        const comp = new Composition();
        if ("title" in composition) comp.title = composition['title'] as string;
        if ("description" in composition) comp.description = composition['description'] as string;
        if ("tempo" in composition) comp.tempo = composition['tempo'] as number;
        if ("events" in composition)
            comp.events = (composition['events'] as IGridItem[]).map((e: IGridItem) => new GridItem(this, e));
        
    }
    download(): Composition {
        const comp = new Composition();
        comp.events = this.notes;        
        
        const data = JSON.stringify(comp);
        const blob = new Blob([data], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'composition.json';
        a.click();
        URL.revokeObjectURL(url);

        return comp;
    }
}

export class PianoRoll {
    domElement: HTMLDivElement;
    grid: Grid;
    sideKeyboard: SideKeyboard;

    constructor() {
        this.domElement = document.createElement('div');
        this.domElement.style.width = '100%';
        this.domElement.style.height = '100%';
        this.domElement.style.display = 'flex';
        this.domElement.style.flexDirection = 'row';

        this.sideKeyboard = new SideKeyboard();
        this.domElement.appendChild(this.sideKeyboard.domElement);

        this.grid = new Grid();
        this.domElement.appendChild(this.grid.domElement);
    }

}

class SideKeyboard {
    domElement: HTMLDivElement;
    keys: HTMLDivElement[] = [];
    noteHeight = 10;

    constructor() {
        this.domElement = document.createElement('div');
        this.domElement.style.position = 'absolute';
        this.domElement.style.width = '50px';
        this.domElement.style.height = '100%';
        this.domElement.style.backgroundColor = '#f0f0f0';
        this.domElement.style.overflow = 'hidden';

        this.redraw();
    }

    redraw() {

        this.domElement.innerHTML = '';

        for (let i = 0; i < keyColors.length * 5; i++) {
            const key = document.createElement('div');
            key.style.position = 'absolute';
            key.style.width = '100%';
            key.style.height = `${this.noteHeight}px`;
            key.style.top = `${i * this.noteHeight}px`;
            key.style.backgroundColor = keyColors[i] === 'white' ? '#fff' : '#000';
            key.style.border = '1px solid #000';
            key.style.boxSizing = 'border-box';
            key.style.textAlign = 'center';
            key.style.lineHeight = `${this.noteHeight}px`;
            key.textContent = notes[i] + (i / notes.length | 0);
            this.domElement.appendChild(key);
        }
    }
}


