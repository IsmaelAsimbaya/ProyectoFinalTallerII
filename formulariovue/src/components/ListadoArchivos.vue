<template>
    <div class="container">
        <h2>Archives</h2>
        <table class="table table-bordered">
            <thead>
            <tr>
                <th>Name</th>
                <th>Mimetype</th>
                <th>size</th>
            </tr>
            </thead>
            <tbody>
              <tr v-for="(item,index) in archives" :key="index">
                
                  <td>{{item.filename}}</td>
                  <td>{{item.mimetype}}</td>
                  <td>{{item.size}}</td>
              </tr>
            </tbody>
        </table>
    </div>
</template>

<script>
import axios from 'axios';

    export default {

        name: 'ListadoArchivos',
       data(){
           return{
               archives: [],
           }
       },mounted(){

            //Recojo todos los archivos que fueron guardados en un array en el servidor y lo que hago es mostrar en una tabla, su nombre, su tipo y su tamaño.
            
            axios.get('/api/archivos').then(resp=>{

                this.archives=resp.data;
                //console.log(resp);
            });
        }
     }
</script>

