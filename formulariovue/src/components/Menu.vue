<template>
    <nav class="navbar navbar-expand-sm bg-dark navbar-dark">
        <div class="container-fluid">
            <ul class="navbar-nav">
            <li class="nav-item">
                <router-link class="nav-link" to="/formulario">Insertar archivo</router-link>
            </li>
            <li class="nav-item">
                <router-link class="nav-link" to="/listadoarchivos">Listado de archivos</router-link>
            </li>
            </ul>
        </div>
    </nav>
</template>

<script>
export default {
    name: "Menu"
}
</script>

<style>

</style>