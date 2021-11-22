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

function submitCastingPriorInfo(form, title="预测结果") {
    axios.get('http://localhost:8080/getcastingcurvefrominput', {Headers:{
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json"
    }}).then(response => {
        generatingCastingChart(response.data.castingCurveValues, title)
        translateData = response.data.castingCurveValues;
        startTime = Date.now();
        _animationStarted = true;
    }).then().catch(
        err => console.log(err))
}

var chartListViewModel = Vue.createApp({
    data(){
        return {
            tabTitles: ['标题一', '标题二', '标题三', '标题四'],
            tabMains: ['内容一', '内容二', '内容三', '内容四'],
            currentIndex:0
        }
    }
}).mount("#tabRow");