function onLoad() {
    echart = echarts.init(document.getElementById("lossChartDiv"));
    // displayLossChart(null)
    taskId = parseInt(getQueryString("taskId"));
    // getTaskByid(taskId);
    getTaskStatusById(taskId);
}

window.onresize = function(){
    resizeEverything();
}

var taskStatusVm = Vue.createApp({
    data() {
        return {
            id: 0,
            status: "training",
            losses: {
                time:[],
                value:[]
            }
        }
    }
}).mount("#lossChartDiv");

function getQueryString(name) {
    let reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
    let r = window.location.search.substr(1).match(reg);
    if (r != null) {
        return decodeURIComponent(r[2]);
    };
    return null;
}


function getTaskStatusById(id) {
    axios.get(baseServerAddr + '/getStatusByTaskId?taskId=' + id, {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        data = response.data;
        
        data.lossDates.forEach(element => {
            taskStatusVm.losses.time.push(element)
        });

        data.losses.forEach(element => {
            taskStatusVm.losses.value.push(element)
        });

        displayLossChart(data)
    }).then().catch(
        err => {
            console.log(err)
            showError(err)
        })
}


function displayLossChart()
{
    var option = {
        xAxis: {
            type: 'category',
            data: taskStatusVm.losses.time
        },
        yAxis: [
            {
                type: 'value',
                min: -2
            },
            {
                type: 'value',
                min: -2
            }

        ],
        series: [
            {
                data: taskStatusVm.losses.value,
                yAxis: 0,
                type: 'line'
            }
            // {
            //     data: [1,1.1,1.2],
            //     yAxis: 2,
            //     type: 'line'
            // }
        ]
    };

    echart.setOption(option)
}

window.setInterval(updateStatus, 2000);

function updateStatus()
{
    getTaskStatusById(taskId)
    // const taskStatusAndLoss = getTaskStatusById(taskStatusVm.id)
    // taskStatusAndLoss.losses.forEach(element => {
    //     taskStatusVm.losses.time.add(losses.time);
    //     taskStatusVm.losses.value.add(losses.value);
    // });
}

function resizeEverything()
{
    echart.resize();
}