formVm = Vue.createApp({
    data() {
        return {
            steelType: "钢种",
            castingWidth: "灌注宽度",
            billetWidth: "铸坯宽度",
            billetThickness: "铸坯厚度",
            mode: "铸机模式",
            targetLV: "目标液位",
            predictButton:"预测"
        }
    }
}).mount('#steelInfoForm')

function submitCastingPriorInfo(form) {
    axios.get('http://localhost:8080/getcastingcurvefrominput', {Headers:{
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json"
    }}).then(response => {
        generatingCastingChart(response.data.castingCurveValues)
    }).then().catch(
        err => console.log(err))
}