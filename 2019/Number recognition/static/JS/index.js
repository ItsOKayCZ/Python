var c;
var ctx;

var WIDTH = 0;
var HEIGHT = 0;

var painter;

window.onload = function(){
  AI();

  c = document.getElementById("canvas");
  ctx = c.getContext("2d");

  WIDTH = c.width = c.clientWidth;
  HEIGHT = c.height = c.clientHeight;

  c.onmousemove = function(e){
    painter.draw(getMousePos(e));
  }

  c.onmousedown = function(e){
    painter.pressed(true, getMousePos(e));
  }
  window.onmouseup = function(e){
    painter.pressed(false, getMousePos(e));
  }

  painter = new Painter();
}

function getMousePos(e){
  var el = c.getBoundingClientRect();
  var x = e.clientX - el.left;
  var y = e.clientY - el.top;
  return {x: x, y: y};
}

class Painter{

  prevPos = {x: undefined, y: undefined};

  color = "black";
  radius = 10;

  mousePressed = false;

  constructor(){
    this.clear();
  }

  draw(pos){

    ctx.lineWidth = this.radius * 2;
    ctx.fillStyle = ctx.strokeStyle = this.color;

    if(this.mousePressed){

      if(this.prevPos.x != undefined){
        ctx.beginPath();
        ctx.moveTo(this.prevPos.x, this.prevPos.y);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
      }
      this.prevPos.x = pos.x;
      this.prevPos.y = pos.y;

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, this.radius, 0, Math.PI * 2);
      ctx.fill();
    }

  }

  pressed(b, pos){


    if(!b && this.mousePressed){
      this.prevPos = {x: undefined, y: undefined};
      this.makeImage();
    }

    this.mousePressed = b;
    this.draw(pos);

  }

  makeImage(){

    var img = new Image();
    img.src = c.toDataURL();
    img.onload = function(){
      var el = document.getElementById("temp");
      var temp = el.getContext("2d");

      // Compress
      temp.drawImage(img, 0, 0, img.width, img.height, 0, 0, el.width, el.height);

      var pixels = temp.getImageData(0, 0, el.width, el.height).data;
      var data = [];
      // Grayscale
      for(var i = 0; i < pixels.length; i += 4){
        data.push(pixels[i + 3]);
      }

      temp.clearRect(0, 0, el.width, el.height);
      predict(data);
    }

  }

  clear(){
    ctx.fillStyle = "white";
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    this.prevPos = {x: undefined, y: undefined};
  }
}