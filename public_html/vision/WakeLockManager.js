class WakeLockManager {
    constructor() {
        this.wakeLock = null;
    }

    async request() {
        if ('wakeLock' in navigator) {
            try {
                this.wakeLock = await navigator.wakeLock.request('screen');
                console.log('Screen Wake Lock is active');
            } catch (err) {
                console.error(`${err.name}, ${err.message}`);
            }
        }
    }

    async release() {
        if (this.wakeLock !== null) {
            await this.wakeLock.release();
            this.wakeLock = null;
            console.log('Screen Wake Lock is released');
        }
    }
}