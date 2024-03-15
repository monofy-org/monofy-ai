class TiledCamera {  

  constructor(width, height, rows, cols, onCameraEvent) {
    this.width = width;
    this.height = height;
    this.rows = rows;
    this.cols = cols;
    this.onCameraEvent = onCameraEvent;     
    this.canvas = document.createElement("canvas");
    this.canvas.id = "framesCanvas";
    this.canvas.style.display = "none";    
    this.canvas.width = cols * width;
    this.canvas.height = rows * height;
    this.currentIndex = 0;  
        this.video = null;
    this.stream = null;
    this.continuous = true;
  }  

  start(continuous=true) {
    // open webcam device in browser
    var _this = this;    
    var video = document.createElement("video");
    video.id = "video";
    video.style.display = "none";
    video.width = 640;
    video.height = 480;
    document.body.appendChild(video);
    this.video = video;
    this.continuous = continuous;
    this.currentIndex = 0;

    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((function (stream) {        
        video.srcObject = stream;
        _this.stream = stream;
        video.play();
        console.log("Started.");
        _this.onCameraEvent({ type: "started" });
        _this.timer = setInterval(_this.updateFrame.bind(_this), 1000);
      }));

      window.addEventListener("beforeunload", () => {
        _this.stop();
      });
  }

  stop() {        
    this.video.pause();
    this.stream.getTracks().forEach(function(track) {
      track.stop();
    });  
    clearInterval(this.timer);
    console.log("Stopped.");
    this.onCameraEvent({ type: "stopped" });
  }

  updateFrame() {       

    if (this.video.readyState !== video.HAVE_ENOUGH_DATA) {
      console.log("Waiting for video to load.");
      return;
    }

    console.log("Capturing frame " + this.currentIndex);
    
    var ctx = this.canvas.getContext("2d");
    const x = (this.currentIndex % this.cols) * this.width;
    const y = Math.floor(this.currentIndex / this.cols) * this.height;
    ctx.drawImage(this.video, x, y, this.width, this.height);

    this.onCameraEvent({ type: "frame", index: this.currentIndex, video: video });

    console.log("Captured frame " + this.currentIndex, x, y);

    this.currentIndex++;
    if (this.currentIndex === this.rows * this.cols) {
      this.currentIndex = 0;
      this.onCameraEvent({ type: "tiles", canvas: this.canvas });      
      if (!this.continuous) {
        this.stop();
      }
      else {
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      }      
    }
  }

}
