document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("engine-select").addEventListener("change", function () {
        this.form.submit();
    });
});

