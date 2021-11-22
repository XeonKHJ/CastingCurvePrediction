formVm = Vue.createApp({
    data() {
        return {
            steelType: "钢种",
            castingWidth: "灌注宽度",
            billetWidth: "铸坯宽度",
            billetThickness: "铸坯厚度",
            mode: "铸机模式",
            targetLV: "目标液位",
            predictButton: "预测"
        }
    }
}).mount('#steelInfoForm')

function submitCastingPriorInfo(form, title = "预测结果") {
    axios.get('http://localhost:8080/getcastingcurvefrominput', {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        if (chartCollectionViewModel.isEmpty) {
            chartCollectionViewModel.chartViewModels.splice(0, 1);
            chartCollectionViewModel.isEmpty = false;
        }
        vmId = hashCode(title);
        var filePaths = title.split('\\');
        var fileName = filePaths[filePaths.length -1]
        var newDataViewModel = {
            chartId:vmId,
            title:fileName,
            data:response.data.castingCurveValues
        }
        chartCollectionViewModel.chartViewModels.push(newDataViewModel);
        generatingCastingChart(newDataViewModel)
        translateData = response.data.castingCurveValues;
        startTime = Date.now();
        _animationStarted = true;
    }).then().catch(
        err => console.log(err))
}

var chartListViewModel = Vue.createApp({
    data() {
        return {
            tabTitles: ['没有数据'],
            tabDatas: [],
            currentIndex: 0,
        }
    }
})

var chartCollectionViewModel = Vue.createApp({
    data() {
        return {
            isEmpty : true,
            chartViewModels: [
                {
                    chartId:0,
                    title: "没数据",
                    data: null,
                    isPredictResult : false,
                    isSelected: true,
                    selectedLiClassName:"tabItemSelected"
                }
            ],
            currentId:0

        }
    }
}).mount("#tabRow");


function clickTab() {
    console.log("Tab" + chartListViewModel.currentIndex + "is clicked");
    selectItem(chartCollectionViewModel.currentId)
}

function hashCode(str, seed = 0) {
    let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
    for (let i = 0, ch; i < str.length; i++) {
        ch = str.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1>>>16), 2246822507) ^ Math.imul(h2 ^ (h2>>>13), 3266489909);
    h2 = Math.imul(h2 ^ (h2>>>16), 2246822507) ^ Math.imul(h1 ^ (h1>>>13), 3266489909);
    return 4294967296 * (2097151 & h2) + (h1>>>0);
};

function selectItem(id)
{
    chartCollectionViewModel.chartViewModels.forEach(element => {
        if(element.chartId == id)
        {
            element.isSelected = true;
        }
        else
        {
            element.isSelected = false;
        }
    });
}