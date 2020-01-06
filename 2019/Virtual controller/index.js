function sendDirection(dir, b){

  var http = new XMLHttpRequest();
  http.open("GET", location.origin + "/changeDirection?dir=" + dir + "&b=" + b);
  http.send();

}

function detect(){
  var up = document.getElementsByClassName("up")[0];
  var down = document.getElementsByClassName("down")[0];
  var left = document.getElementsByClassName("left")[0];
  var right = document.getElementsByClassName("right")[0];
  var interact = document.getElementsByClassName("interact")[0];
  var jump = document.getElementsByClassName("jump")[0];

  if(navigator.userAgent.match(/Android/i) ||
     navigator.userAgent.match(/webOS/i) ||
     navigator.userAgent.match(/iPhone/i) ||
     navigator.userAgent.match(/iPad/i) ||
     navigator.userAgent.match(/iPod/i) ||
     navigator.userAgent.match(/BlackBerry/i) ||
     navigator.userAgent.match(/Windows Phone/i)){

    up.ontouchstart = function(){sendDirection("up", true);}
    up.ontouchend = function(){sendDirection("up", false);}

    down.ontouchstart = function(){sendDirection("down", true);}
    down.ontouchend = function(){sendDirection("down", false);}

    left.ontouchstart = function(){sendDirection("left", true);}
    left.ontouchend = function(){sendDirection("left", false);}

    right.ontouchstart = function(){sendDirection("right", true);}
    right.ontouchend = function(){sendDirection("right", false);}

    interact.ontouchstart = function(){sendDirection("interact");}
    jump.ontouchstart = function(){sendDirection("jump");}
  } else {
    up.onmousedown = function(){sendDirection("up", true);}
    up.onmouseup = function(){sendDirection("up", false);}

    down.onmousedown = function(){sendDirection("down", true);}
    down.onmouseup = function(){sendDirection("down", false);}

    left.onmousedown = function(){sendDirection("left", true);}
    left.onmouseup = function(){sendDirection("left", false);}

    right.onmousedown = function(){sendDirection("right", true);}
    right.onmouseup = function(){sendDirection("right", false);}

    interact.onmousedown = function(){sendDirection("interact");}
    jump.onmousedown = function(){sendDirection("jump");}
  }
}
window.onload = detect;
window.onmousedown = function(){
  document.documentElement.requestFullscreen();
}