function uuid() {
  return Math.random().toString(16).slice(2);
}

export class Folder {
  created: Date;
  updated: Date | null = null;
  constructor(
    protected name: string,
    public parentId?: string,
    public readonly id = uuid()
  ) {
    this.created = new Date();
    this.updated = new Date();
  }
}

export class StorageItem {
  public readonly id: string;
  created: Date;
  updated: Date | null = null;
  constructor(
    public readonly name: string,
    public readonly data: unknown,
    public readonly parentId: string | null = null
  ) {
    this.id = uuid();
    this.created = new Date();
    this.updated = new Date();
  }
}

export class Storage {
  private db: IDBDatabase | null = null;

  constructor(public readonly storeName: string) {

    console.log("Creating storage", storeName);

    this.storeName = storeName;
    const openRequest = indexedDB.open(this.storeName);

    openRequest.onupgradeneeded = () => {
      this.db = openRequest.result;
      if (!this.db.objectStoreNames.contains(this.storeName)) {
        const objectStore = this.db.createObjectStore(this.storeName, {
          keyPath: "id",
          autoIncrement: true,
        });

        // Create an index to search folders by parentId.
        objectStore.createIndex("id", "id", { unique: true });
        objectStore.createIndex("parentId", "parentId", { unique: false });
      }
    };

    openRequest.onsuccess = () => {
      this.db = openRequest.result;

      console.log("Database opened", this.db);

      // Check if the object store has been created
      if (!this.db.objectStoreNames.contains(this.storeName)) {
        console.error(
          "Error",
          "Failed to create object store " + this.storeName
        );
      }
    };

    openRequest.onerror = () => {
      console.error("Error", openRequest.error);
    };
  }

  async add(item: StorageItem): Promise<void> {
    const transaction = this.db?.transaction(this.storeName, "readwrite");
    const store = transaction?.objectStore(this.storeName);
    await store?.add(item);
  }

  async createFolder(name: string, parentId: string): Promise<Folder> {
    const folder = new Folder(name, parentId);
    return this.addFolder(folder);
  }

  async addFolder(folder: Folder): Promise<Folder> {
    const transaction = this.db?.transaction(this.storeName, "readwrite");
    const store = transaction?.objectStore(this.storeName);
    await store?.add(folder);
    return folder;
  }

  async delete(id: string): Promise<void> {
    const transaction = this.db?.transaction(this.storeName, "readwrite");
    const store = transaction?.objectStore(this.storeName);

    // delete all children
    const index = store?.index("parentId");
    const request = index?.getAll(id);
    if (!request) {
      console.error("Error", "Failed to get items for parent ID " + id);
    } else {
      request.onsuccess = () => {
        const children = request.result;
        for (const child of children) {
          this.delete(child.id);
        }
      };
    }

    store?.delete(id);
  }

  async get(name: string): Promise<StorageItem | undefined> {
    const transaction = this.db?.transaction(this.storeName, "readonly");
    const store = transaction?.objectStore(this.storeName);
    const request = store?.get(name);

    return new Promise((resolve, reject) => {
      request?.addEventListener("success", function () {
        resolve(request?.result);
      });
      request?.addEventListener("error", function () {
        reject(request?.error);
      });
    });
  }

  async getFolderItems(id: string | null): Promise<StorageItem[]> {
    const transaction = this.db?.transaction(this.storeName, "readonly");
    const store = transaction?.objectStore(this.storeName);
    const index = store?.index("parentId");
    const request = index?.getAll(id);

    return new Promise((resolve, reject) => {
      request?.addEventListener("success", function () {
        resolve(request?.result);
      });
      request?.addEventListener("error", function () {
        reject(request?.error);
      });
    });
  }
}
