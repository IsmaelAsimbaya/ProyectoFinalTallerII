<template>
    <div>
        <div class="container">
            <form @submit.prevent="formulario()" enctype="multipart/form-data">
                <div class="row form-group">
                    <label>Seleccione un archivo:</label>
                    <input type="file" class="form-control" name="archivo" @change="buscarArchivo"/>
                </div>
                <div class="row form-group">
                    <button class="btn btn-secondary">Create</button>
                </div>
            </form>
            
        </div>
        <div class="alert alert-danger" ref="message" hidden>
            <p>Only files or images are allowed.</p>
        </div>
     </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'CreateUser',
  data() {
    return {
        
        fichero:null
    }
  },
  methods: {
     
      //Con éste método lo que hacemos es recoger el archivo pasado en el input file y comprobamos si cumple con la condición de tener el mismo tipo/extensión que los guardados en el array, si es así entonces la variable fichero puede recoger el valor del input, si no lo es entonces le aparece un mensaje de error y no se podría enviar nada al servidor.

        buscarArchivo(event){
            
           var file=event.target.files[0];
           const allowedTypes=["application/pdf","application/docx","application/txt","image/jpg","image/jpeg","image/png","image/gif"];

           if(allowedTypes.includes(file.type)){

                this.fichero=file;
                this.$refs.message.setAttribute("hidden",true);
           }else{

                this.$refs.message.removeAttribute("hidden");
           }
            
        },
        
        //Convertimos dicho valor a un objeto formData y se lo enviamos al servidor, si se cumple esto con éxtio entonces nos redireccione a la página de listado mostrando todos los archivos guardados.

        formulario(){

            const formdata=new FormData();
            formdata.append('file',this.fichero);

            axios.post('/api/formulario',formdata).then(resp=>{
                
                console.log(resp);
                 this.$router.push("/listadoarchivos");
            });
        }
  }
}
</script>