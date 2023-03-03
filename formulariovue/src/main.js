import Vue from 'vue';
import App from './App.vue';
import { BootstrapVue } from 'bootstrap-vue';
import VueRouter from 'vue-router';
import Formulario from './components/Formulario.vue';
import ListadoArchivos from './components/ListadoArchivos.vue';


Vue.config.productionTip = false

Vue.use(VueRouter);

const routes=[

  {path:"/formulario", component: Formulario},
  {path:"/listadoarchivos", component: ListadoArchivos}

]

const router=new VueRouter({

  routes,
  mode:"history"
});

new Vue({
  router,
  render: h => h(App),
}).$mount('#app')

Vue.use(BootstrapVue);
