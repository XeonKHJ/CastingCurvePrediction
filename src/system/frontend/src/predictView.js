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
        if(chartListViewModel.tabDatas.length == 0)
        {
            chartListViewModel.tabTitles.splice(0, 1);
        }
        chartListViewModel.tabTitles.push(title);
        chartListViewModel.tabDatas.push(response.data.castingCurveValues)
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
            tabTitles: ['没有数据'],
            tabDatas: [],
            currentIndex:0,
        }
    }
}).mount("#tabRow");

function clickTab()
{
    console.log("Tab" + chartListViewModel.currentIndex + "is clicked");
    
}