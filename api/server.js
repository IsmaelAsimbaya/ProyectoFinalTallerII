//Creamos las dependencias que vamos a usar
const express = require('express');
const path = require('path');
const multer=require('multer');

const app = express(),
    bodyParser = require("body-parser");
    port = 3070;


//Con este filtro lo que hacemos es que, dentro de los tipos de extensión de los archivos declaramos, el archivo que subamos al servidor deberá ser de alguno de estos tipos, sino entonces saltará un error diciendo que el tipo de archivo es incorrecto.

const fileFilter=function(req,file,cb){

    const allowedTypes=["image/jpg","image/jpeg","image/png","image/gif","image/webp"];

    if(!allowedTypes.includes(file.mimetype)){

      const error=new Error("wrong file type");
      error.code="LIMIT_FILE_TYPES";
      return cb(error,false);
    }

    cb(null,true);
}

//Con la dependencia multer lo que hacemos es asignar a qué ruta irán guardados los archivos y qué tipo de nombre tendrá (en este caso he preferido que conserve el nombre original).

const storage=multer.diskStorage({
  destination:'../FRDeepLearning/knn_examples/test',
  filename:(req,file,cb)=>{
    cb(null,file.originalname);
  }
})

//A la dependencia le pasamos tanto el almacenamiento/nombre de archivo como el filtro del cual tendrá que pasar dicho archivo para poder ser guardado en el servidor.

const files=[];

const upload=multer({storage,fileFilter});

app.use(bodyParser.json());

//Lo que le decimos aquí es que nos redirrecione al index.html de nuestro proyecto de vue.
app.use(express.static(path.join(__dirname, '../formulariovue/dist')));

//Enviamos el array de todos los archivos que son añadidos en dicho array cada vez que se suven al servidor.

app.get('/api/archivos', (req, res) => {
  console.log('api/archivos called!')
  res.json(files);
});

//Cuando llamemos al método post debemos pasarle por parámetro un archivo y éste es pasado por la dependencia de multer haciendo lo declarado en sus configuraciones arriba (lo de storage y filtername).

app.post('/api/formulario',upload.single('file'),(req, res) => {

  files.push(req.file);
  res.json({file: req.file});
});

//Con esta función hacemos el control del filtro, sino lo cumple, entonces devuelve el error 422 con un mensaje.

app.use(function(err,req,res,next){

    if(err.code==="LIMIT_FILE_TYPES"){

      res.status(422).json({error: "Only images or files are allowed"});
    }
})


//Lo que hacemos aquí es redireccionar al index.html de la carpeta build (es la carpeta que se crea cada vez que se compila el proyecto vue pudiendo así poder tener comunicación entre la api y el proyecto).

app.get('/', (req,res) => {
  res.sendFile(path.join(__dirname, '../formulariovue/build/index.html'));
});

//Es el puerto por el que escucha el servidor (es el declarado arriba, el 3080).

app.listen(port, () => {
    console.log(`Server listening on the port::${port}`);
});