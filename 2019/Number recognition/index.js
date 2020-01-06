const express = require("express");
const app = express();

const PORT = 8080;

app.use(express.static("static"));

app.listen(PORT, function(){
  console.log("Listening on localhost:" + PORT);
});