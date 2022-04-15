var dialogViewModel = Vue.createApp({
    data() {
        return {
            title: "dialog",
            message: "nothing",
            display: false
        }
    }
}).mount("#dialogDiv")

function showError(message) {
    var galleryModal = new bootstrap.Modal(document.getElementById('dialogDiv'), {
        keyboard: false
    });
    dialogViewModel.title = "❌错误"
    dialogViewModel.message = message
    galleryModal.show();
}

