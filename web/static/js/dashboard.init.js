//function getChartColorsArray(r){r=$(r).attr("data-colors");return(r=JSON.parse(r)).map(function(r){r=r.replace(" ","");if(-1==r.indexOf("--"))return r;r=getComputedStyle(document.documentElement).getPropertyValue(r);return r||void 0})}var minichart1Colors=getChartColorsArray("#mini-chart1"),options={series:[{data:[2,10,18,22,36,15,47,75,65,19,14,2,47,42,15]}],chart:{type:"line",height:50,sparkline:{enabled:!0}},colors:minichart1Colors,stroke:{curve:"smooth",width:2},tooltip:{fixed:{enabled:!1},x:{show:!1},y:{title:{formatter:function(r){return""}}},marker:{show:!1}}},chart=new ApexCharts(document.querySelector("#mini-chart1"),options);chart.render();var minichart2Colors=getChartColorsArray("#mini-chart2"),options={series:[{data:[15,42,47,2,14,19,65,75,47,15,42,47,2,14,12]}],chart:{type:"line",height:50,sparkline:{enabled:!0}},colors:minichart2Colors,stroke:{curve:"smooth",width:2},tooltip:{fixed:{enabled:!1},x:{show:!1},y:{title:{formatter:function(r){return""}}},marker:{show:!1}}};(chart=new ApexCharts(document.querySelector("#mini-chart2"),options)).render();var minichart3Colors=getChartColorsArray("#mini-chart3"),options={series:[{data:[47,15,2,67,22,20,36,60,60,30,50,11,12,3,8]}],chart:{type:"line",height:50,sparkline:{enabled:!0}},colors:minichart3Colors,stroke:{curve:"smooth",width:2},tooltip:{fixed:{enabled:!1},x:{show:!1},y:{title:{formatter:function(r){return""}}},marker:{show:!1}}};(chart=new ApexCharts(document.querySelector("#mini-chart3"),options)).render();var minichart4Colors=getChartColorsArray("#mini-chart4"),options={series:[{data:[12,14,2,47,42,15,47,75,65,19,14,2,47,42,15]}],chart:{type:"line",height:50,sparkline:{enabled:!0}},colors:minichart4Colors,stroke:{curve:"smooth",width:2},tooltip:{fixed:{enabled:!1},x:{show:!1},y:{title:{formatter:function(r){return""}}},marker:{show:!1}}};(chart=new ApexCharts(document.querySelector("#mini-chart4"),options)).render();var piechartColors=getChartColorsArray("#wallet-balance"),options={series:[35,70,15],chart:{width:227,height:227,type:"pie"},labels:["Ethereum","Bitcoin","Litecoin"],colors:piechartColors,stroke:{width:0},legend:{show:!1},responsive:[{breakpoint:480,options:{chart:{width:200}}}]};(chart=new ApexCharts(document.querySelector("#wallet-balance"),options)).render();var radialchartColors=getChartColorsArray("#invested-overview"),options={chart:{height:270,type:"radialBar",offsetY:-10},plotOptions:{radialBar:{startAngle:-130,endAngle:130,dataLabels:{name:{show:!1},value:{offsetY:10,fontSize:"18px",color:void 0,formatter:function(r){return r+"%"}}}}},colors:[radialchartColors[0]],fill:{type:"gradient",gradient:{shade:"dark",type:"horizontal",gradientToColors:[radialchartColors[1]],shadeIntensity:.15,inverseColors:!1,opacityFrom:1,opacityTo:1,stops:[20,60]}},stroke:{dashArray:4},legend:{show:!1},series:[80],labels:["Series A"]};(chart=new ApexCharts(document.querySelector("#invested-overview"),options)).render();var barchartColors=getChartColorsArray("#market-overview"),options={series:[{name:"Profit",data:[12.45,16.2,8.9,11.42,12.6,18.1,18.2,14.16,11.1,8.09,16.34,12.88]},{name:"Loss",data:[-11.45,-15.42,-7.9,-12.42,-12.6,-18.1,-18.2,-14.16,-11.1,-7.09,-15.34,-11.88]}],chart:{type:"bar",height:400,stacked:!0,toolbar:{show:!1}},plotOptions:{bar:{columnWidth:"20%"}},colors:barchartColors,fill:{opacity:1},dataLabels:{enabled:!1},legend:{show:!1},yaxis:{labels:{formatter:function(r){return r.toFixed(0)+"%"}}},xaxis:{categories:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],labels:{rotate:-90}}};(chart=new ApexCharts(document.querySelector("#market-overview"),options)).render();var vectormapColors=getChartColorsArray("#sales-by-locations");$("#sales-by-locations").vectorMap({map:"world_mill_en",normalizeFunction:"polynomial",hoverOpacity:.7,hoverColor:!1,regionStyle:{initial:{fill:"#e9e9ef"}},markerStyle:{initial:{r:9,fill:vectormapColors,"fill-opacity":.9,stroke:"#fff","stroke-width":7,"stroke-opacity":.4},hover:{stroke:"#fff","fill-opacity":1,"stroke-width":1.5}},backgroundColor:"transparent",markers:[{latLng:[41.9,12.45],name:"USA"},{latLng:[12.05,-61.75],name:"Russia"},{latLng:[1.3,103.8],name:"Australia"}]});

function getChartColorsArray(r) {
    r = $(r).attr("data-colors");
    return (r = JSON.parse(r)).map(function (r) {
        r = r.replace(" ", "");
        if (-1 === r.indexOf("--")) return r;
        r = getComputedStyle(document.documentElement).getPropertyValue(r);
        return r || void 0;
    });
}

// // Mini Chart 1
// var minichart1Colors = getChartColorsArray("#mini-chart1");
// var options = {
//     series: [{
//         data: [2, 10, 18, 22, 36, 15, 47, 75, 65, 19, 14, 2, 47, 42, 15]
//     }],
//     chart: {
//         type: "line",
//         height: 50,
//         sparkline: {
//             enabled: true
//         }
//     },
//     colors: minichart1Colors,
//     stroke: {
//         curve: "smooth",
//         width: 2
//     },
//     tooltip: {
//         fixed: {
//             enabled: false
//         },
//         x: {
//             show: false
//         },
//         y: {
//             title: {
//                 formatter: function (r) {
//                     return "";
//                 }
//             }
//         },
//         marker: {
//             show: false
//         }
//     }
// };
// var chart = new ApexCharts(document.querySelector("#mini-chart1"), options);
// chart.render();

// // Mini Chart 2
// var minichart2Colors = getChartColorsArray("#mini-chart2");
// var options = {
//     series: [{
//         data: [15, 42, 47, 2, 14, 19, 65, 75, 47, 15, 42, 47, 2, 14, 12]
//     }],
//     chart: {
//         type: "line",
//         height: 50,
//         sparkline: {
//             enabled: true
//         }
//     },
//     colors: minichart2Colors,
//     stroke: {
//         curve: "smooth",
//         width: 2
//     },
//     tooltip: {
//         fixed: {
//             enabled: false
//         },
//         x: {
//             show: false
//         },
//         y: {
//             title: {
//                 formatter: function (r) {
//                     return "";
//                 }
//             }
//         },
//         marker: {
//             show: false
//         }
//     }
// };
// chart = new ApexCharts(document.querySelector("#mini-chart2"), options);
// chart.render();

// // Mini Chart 3
// var minichart3Colors = getChartColorsArray("#mini-chart3");
// var options = {
//     series: [{
//         data: [47, 15, 2, 67, 22, 20, 36, 60, 60, 30, 50, 11, 12, 3, 8]
//     }],
//     chart: {
//         type: "line",
//         height: 50,
//         sparkline: {
//             enabled: true
//         }
//     },
//     colors: minichart3Colors,
//     stroke: {
//         curve: "smooth",
//         width: 2
//     },
//     tooltip: {
//         fixed: {
//             enabled: false
//         },
//         x: {
//             show: false
//         },
//         y: {
//             title: {
//                 formatter: function (r) {
//                     return "";
//                 }
//             }
//         },
//         marker: {
//             show: false
//         }
//     }
// };
// chart = new ApexCharts(document.querySelector("#mini-chart3"), options);
// chart.render();

// // Mini Chart 4
// var minichart4Colors = getChartColorsArray("#mini-chart4");
// var options = {
//     series: [{
//         data: [12, 14, 2, 47, 42, 15, 47, 75, 65, 19, 14, 2, 47, 42, 15]
//     }],
//     chart: {
//         type: "line",
//         height: 50,
//         sparkline: {
//             enabled: true
//         }
//     },
//     colors: minichart4Colors,
//     stroke: {
//         curve: "smooth",
//         width: 2
//     },
//     tooltip: {
//         fixed: {
//             enabled: false
//         },
//         x: {
//             show: false
//         },
//         y: {
//             title: {
//                 formatter: function (r) {
//                     return "";
//                 }
//             }
//         },
//         marker: {
//             show: false
//         }
//     }
// };
// chart = new ApexCharts(document.querySelector("#mini-chart4"), options);
// chart.render();

// Pie Chart
var piechartColors = getChartColorsArray("#wallet-balance");
var options = {
    series: [35, 70, 15],
    chart: {
        width: 227,
        height: 227,
        type: "pie"
    },
    labels: ["Ethereum", "Bitcoin", "Litecoin"],
    colors: piechartColors,
    stroke: {
        width: 0
    },
    legend: {
        show: false
    },
    responsive: [{
        breakpoint: 480,
        options: {
            chart: {
                width: 200
            }
        }
    }]
};
chart = new ApexCharts(document.querySelector("#wallet-balance"), options);
chart.render();

// Radial Chart
var radialchartColors = getChartColorsArray("#invested-overview");
var options = {
    chart: {
        height: 270,
        type: "radialBar",
        offsetY: -10
    },
    plotOptions: {
        radialBar: {
            startAngle: -130,
            endAngle: 130,
            dataLabels: {
                name: {
                    show: false
                },
                value: {
                    offsetY: 10,
                    fontSize: "18px",
                    color: void 0,
                    formatter: function (r) {
                        return r + "%";
                    }
                }
            }
        }
    },
    colors: [radialchartColors[0]],
    fill: {
        type: "gradient",
        gradient: {
            shade: "dark",
            type: "horizontal",
            gradientToColors: [radialchartColors[1]],
            shadeIntensity: 0.15,
            inverseColors: false,
            opacityFrom: 1,
            opacityTo: 1,
            stops: [20, 60]
        }
    },
    stroke: {
        dashArray: 4
    },
    legend: {
        show: false
    },
    series: [80],
    labels: ["Series A"]
};
chart = new ApexCharts(document.querySelector("#invested-overview"), options);
chart.render();

// Bar Chart
var barchartColors = getChartColorsArray("#market-overview");
var options = {
    series: [{
        name: "Profit",
        data: [12.45, 16.2, 8.9, 11.42, 12.6, 18.1, 18.2, 14.16, 11.1, 8.09, 16.34, 12.88]
    }, {
        name: "Loss",
        data: [-11.45, -15.42, -7.9, -12.42, -12.6, -18.1, -18.2, -14.16, -11.1, -7.09, -15.34, -11.88]
    }],
    chart: {
        type: "bar",
        height: 400,
        stacked: true,
        toolbar: {
            show: false
        }
    },
    plotOptions: {
        bar: {
            columnWidth: "20%"
        }
    },
    colors: barchartColors,
    fill: {
        opacity: 1
    },
    dataLabels: {
        enabled: false
    },
    legend: {
        show: false
    },
    yaxis: {
        labels: {
            formatter: function (r) {
                return r.toFixed(0) + "%";
            }
        }
    },
    xaxis: {
        categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        labels: {
            rotate: -90
        }
    }
};
chart = new ApexCharts(document.querySelector("#market-overview"), options);
chart.render();

// Vector Map
var vectormapColors = getChartColorsArray("#sales-by-locations");
$("#sales-by-locations").vectorMap({
    map: "world_mill_en",
    normalizeFunction: "polynomial",
    hoverOpacity: 0.7,
    hoverColor: false,
    regionStyle: {
        initial: {
            fill: "#e9e9ef"
        }
    },
    markerStyle: {
        initial: {
            r: 9,
            fill: vectormapColors,
            "fill-opacity": 0.9,
            stroke: "#fff",
            "stroke-width": 7,
            "stroke-opacity": 0.4
        },
        hover: {
            stroke: "#fff",
            "fill-opacity": 1,
            "stroke-width": 1.5
        }
    },
    backgroundColor: "transparent",
    markers: [{
        latLng: [41.9, 12.45],
        name: "USA"
    }, {
        latLng: [12.05, -61.75],
        name: "Russia"
    }, {
        latLng: [1.3, 103.8],
        name: "Australia"
    }]
});
//----------------------------------------------



function getChartColorsArray(r) {
    r = $(r).attr("data-colors");
    return (r = JSON.parse(r)).map(function (r) {
        r = r.replace(" ", "");
        if (-1 == r.indexOf("--")) return r;
        r = getComputedStyle(document.documentElement).getPropertyValue(r);
        return r || void 0;
    });
}

function createChart(selector, historicalData, forecastData) {
    var colors = getChartColorsArray(selector);
    var options = {
        series: [
            {
                name: "Forecast Data",
                data: forecastData,
                color: "#a3c585",
                dashArray: 5 // 预测数据使用虚线
            },
            {
                name: "Historical Data",
                data: historicalData,
                color: colors[0]
            }
        ],
        annotations: {
            // xaxis: [
            //     {
            //         x: Math.max(...historicalData.map((_, i) => i)),
            //         borderColor: "#ff0000",
            //         label: {
            //             // text: "End of Historical Data",
            //             style: {
            //                 color: "#ff0000",
            //                 background: "#fff"
            //             }
            //         }
            //     }
            // ]
        },
        chart: {
            type: "line",
            height: 50,
            sparkline: { enabled: true }
        },
        stroke: {
            curve: "smooth",
            width: 2
        },
        tooltip: {
            fixed: { enabled: false },
            x: { show: false },
            y: { title: { formatter: function () { return ""; } } },
            marker: { show: false }
        }
    };

    var chart = new ApexCharts(document.querySelector(selector), options);
    chart.render();
}

// 创建多个 mini-chart
// createChart("#mini-chart1", [2, 10, 18, 22, 36, 15, 47, 75], [65, 19, 14, 2, 47, 42, 15]);
createChart("#mini-chart1", [162.3, 198.7, 176.4, 215.2, 243.6, 244.1, 298.5, 267.3, 289.2, 279.8, 255.7, 224.9], [141.0, 181.3, 169.8, 206.4, 245.2, 264.7, 288.4, 255.1, 279.6, 286.3]);
createChart("#mini-chart2", [15, 42, 47, 2, 14, 19, 65, 75], [47, 15, 42, 47, 2, 14, 12]);
createChart("#mini-chart3", [47, 15, 2, 67, 22, 20, 36, 60], [60, 30, 50, 11, 12, 3, 8]);
createChart("#mini-chart4", [12, 14, 2, 47, 42, 15, 47, 75], [65, 19, 14, 2, 47, 42, 15]);
