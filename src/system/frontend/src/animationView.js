var _animationStarted = true;

var AnimationController = {
    isPlaying : true,
    height : 2000,
    width : 1000,
    webGlContext : null,
    session : null,
    renderFrame(gl){
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    
        bundles = initVertices(gl);
    
        // set global scale matrix
        var u_GlobalModelMatrix = gl.getUniformLocation(gl.program, 'u_GlobalModelMatrix');
        var globalModelMatrix = new Matrix4();
        globalModelMatrix.translate(0, 0.3, 0);
        globalModelMatrix.scale(0.4, 0.4, 1);
        gl.uniformMatrix4fv(u_GlobalModelMatrix, false, globalModelMatrix.elements)
    
        var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
        drawAnimObjBundle(gl, a_Position, bundles);
    },
    createSession(data, startTime){
        return {
            data : data,
            startTime : startTime
        }
    },
    startAnimation(data){
        this.session = this.createSession(data, Date.now())
    },
    stopAnimation(){
        this.session = null
    }
}

var AnimObjHelper = {
    AnimObj(vertices, colors, drawMethod, verticeSize, borderWidth = 0, borderColor = null) {
        return {
            vertices: vertices,
            drawMethod: drawMethod,
            verticeSize: verticeSize,
            colors: colors,
            borderWidth: borderWidth,
            borderColor: null,
            buffer: null
        }
    },
    AnimObjBundle(objList, translationMatrix = [0, 0, 0], scaleMatrix = [0, 0, 0]) {
        return {
            objects: objList,
            transMatrix: translationMatrix,
            scaleMatrix: scaleMatrix
        }
    },
    
}

var GLOBAL_VSHADER_SOURCE =
    'attribute vec4 a_Position;\n' +
    'uniform mat4 u_ModelMatrix; \n' +
    'uniform mat4 u_GlobalModelMatrix; \n' +
    'void main() {\n' +
    '  gl_Position = u_GlobalModelMatrix * u_ModelMatrix * a_Position;\n' +
    '}\n';

// Fragment shader program
var GLOBAL_FSHADER_SOURCE =
    'precision mediump float;\n' +
    'uniform vec4 u_Color;\n' +
    'void main() {\n' +
    '  gl_FragColor = u_Color;\n' +
    '}\n';

function drawAnimation() {
    // Init webgl context.
    var canvas = document.getElementById('animationCanvas');
    var canvasDiv = document.getElementById('animationDiv');
    var gl = getWebGLContext(canvas);
    canvas.height = canvasDiv.clientHeight;
    canvas.width = canvasDiv.clientWidth;

    // Initializev shaders
    if (!initShaders(gl, GLOBAL_VSHADER_SOURCE, GLOBAL_FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

    var tick = function () {
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        height = canvasDiv.clientHeight;
        width = canvasDiv.clientWidth;
        AnimationController.renderFrame(gl);
        if (_animationStarted) {
            requestAnimationFrame(tick, canvas); // Request that the browser ?calls tick
        }
    };
    tick();
}


var height = 2000;
var width = 1000;
var offset = 0;

function reverseVertices(verticesIn) {
    var verticesOut = new Float32Array(
        verticesIn.length
    );
    for (var i = 0; i < verticesIn.length; i++) {
        if (i % 2) {
            verticesOut[i] = verticesIn[i];
        }
        else {
            verticesOut[i] = -1 * verticesIn[i];
        }
    }
    return verticesOut;
}

function a2r(angle) {
    return (angle / 180 * Math.PI);
}

function initVertices(gl) {
    var tan80 = Math.tan(a2r(70));
    var sin80 = Math.sin(a2r(70));
    var tan60 = Math.tan(a2r(60));
    var borderThinkness = 3;
    const tudishParams = {
        borderHeight : 400,
        width : 1500,
        height : 250
    }
    var tudishLeftVertices = new Float32Array(
        [
            -tudishParams.width / width, (tudishParams.height * tan80) / height,
            -(tudishParams.width - ((tudishParams.borderHeight / sin80))) / width, (tudishParams.height * tan80) / height,
            -(tudishParams.width - (tudishParams.height)) / width, 0,
            (-(tudishParams.width - ((tudishParams.borderHeight / sin80)) - (tudishParams.height * tan80 - tudishParams.borderHeight) / tan80)) / width, tudishParams.borderHeight / height,
            -50 / width, 0,
            -50 / width, tudishParams.borderHeight / height
        ]
    );

    var tudishRightVertices = reverseVertices(tudishLeftVertices);

    var tudishLeftInsideVertices = new Float32Array(
        [
            (-(tudishParams.width) + borderThinkness / sin80) / width, ((tudishParams.height * tan80) - borderThinkness) / height,
            (-(tudishParams.width - ((tudishParams.borderHeight / sin80))) - borderThinkness / sin80) / width, ((tudishParams.height * tan80) - borderThinkness) / height,
            (-(tudishParams.width - (tudishParams.height)) + borderThinkness / sin80) / width, (0 + borderThinkness) / height,
            ((-(tudishParams.width - ((tudishParams.borderHeight / sin80)) - (tudishParams.height * tan80 - tudishParams.borderHeight) / tan80)) - borderThinkness) / width, (tudishParams.borderHeight - borderThinkness) / height,
            (-50 - borderThinkness) / width, (0 + borderThinkness) / height,
            (-50 - borderThinkness) / width, (tudishParams.borderHeight - borderThinkness) / height
        ]
    );

    var tudishRightInsideVertices = reverseVertices(tudishLeftInsideVertices);

    const stoperParams = {
        // 不包含下面半圆的塞棒长度
        height:280,
        width:127,
        // 西方扇形部分的圆半径
        r:10
    }

    var sin60 = Math.sin(60 / 180 * Math.PI);
    var sin30 = Math.sin(30 / 180 * Math.PI);
    var stoperVertices = new Float32Array([
        -(stoperParams.width / 2) / width, 300 / height,
        (stoperParams.width / 2) / width, 300 / height,
        -(stoperParams.width / 2) / width, 20 / height,
        stoperParams.width / 2 / width, 20 / height,
        -47.625 / width, 10 / height,
        47.625 / width, 10 / height
    ]);

    var stoperVerticesInside = new Float32Array([
        (-63.5 + borderThinkness) / width, (300 - borderThinkness) / height,
        (63.5 - borderThinkness) / width, (300 - borderThinkness) / height,
        (-63.5 + borderThinkness) / width, (20 + borderThinkness) / height,
        (63.5 - borderThinkness) / width, (20 + borderThinkness) / height,
        (-47.625 + borderThinkness) / width, (10 + borderThinkness * sin60 * sin30) / height,
        (47.625 - borderThinkness) / width, (10 + borderThinkness * sin60 * sin30) / height
    ]);

    var coolingPipeLeftVertices = new Float32Array([
        -40 / width, 0 / height,
        -120 / width, 0 / height,
        -120 / width, -44 / height,
        -110 / width, -44 / height,
        -90 / width, -400 / height,
        -90 / width, -700 / height,
        -32.5 / width, -700 / height
    ]);
    var coolingPipeRightVertices = reverseVertices(coolingPipeLeftVertices);

    var coolingPipeLeftVerticesInside = new Float32Array([
        (-40 - borderThinkness) / width, (0 - borderThinkness) / height,
        (-120 + borderThinkness) / width, (0 - borderThinkness) / height,
        (-120 + borderThinkness) / width, (-44 + borderThinkness) / height,
        (-110 + ((Math.sqrt(Math.pow(356, 2) + Math.pow(10, 2)) - 10) * borderThinkness / 356)) / width, (-44 + borderThinkness) / height,
        (-90 + borderThinkness) / width, (-400 + 10 / 356 * borderThinkness) / height,
        (-90 + borderThinkness) / width, (-700 + borderThinkness) / height,
        (-32.5 - borderThinkness) / width, (-700 + borderThinkness) / height
    ]);

    var coolingPipeRightVerticesInside = reverseVertices(coolingPipeLeftVerticesInside);

    var coolingPipeBottomLeftVertices = new Float32Array([
        -90 / width, -910 / height,
        -90 / width, -(800 - 100) / height,
        -32.5 / width, -(800 - 100) / height,
        -32.5 / width, -(910 - 100 + 50) / height,
        0, -(910 - 100 + 50) / height,
        0, -910 / height
    ])

    var coolingPipeBottomLeftVerticesInside = new Float32Array([
        (-90 + borderThinkness) / width, (-910 + borderThinkness) / height,
        (-90 + borderThinkness) / width, -(800 - 100 - borderThinkness) / height,
        (-32.5 - borderThinkness) / width, -(800 - 100 - borderThinkness) / height,
        (-32.5 - borderThinkness) / width, -(910 - 100 + 50 + borderThinkness) / height,
        0, -(910 - 100 + 50 + borderThinkness) / height,
        0, (-910 + borderThinkness) / height
    ])

    var coolingPipeBottomRightVertices = reverseVertices(coolingPipeBottomLeftVertices);
    var coolingPipeBottomRightVerticesInside = reverseVertices(coolingPipeBottomLeftVerticesInside)

    var coolingPipeBreachVertices = new Float32Array([
        -32.5 / width, -700 / height,
        32.5 / width, -700 / height,
        32.5 / width, -810 / height,
        -32.5 / width, -810 / height
    ])

    var coolingPipeBreachVerticesInside = new Float32Array([
        (-32.5 + borderThinkness) / width, (-700 - borderThinkness) / height,
        (32.5 - borderThinkness) / width, (-700 - borderThinkness)  / height,
        (32.5 - borderThinkness) / width, (-810 + borderThinkness) / height,
        (-32.5 + borderThinkness) / width, (-810 + borderThinkness) / height
    ])

    var stoperBottomVertices = new Float32Array(104);
    stoperBottomVertices[0] = 0;
    stoperBottomVertices[1] = (47.625 + 10) / height;
    index = 2;
    r = 47.625 * Math.sqrt(2);
    for (i = -40; i <= -10; i++) {
        stoperBottomVertices[index] = r * Math.cos(i * 2 * Math.PI / 100) / width;
        stoperBottomVertices[index + 1] = (r * Math.sin(i * 2 * Math.PI / 100) + 47.625 + 10) / height;
        index += 2;
    }

    var stoperBottomVerticesInside = new Float32Array(104);
    stoperBottomVerticesInside[0] = 0;
    stoperBottomVerticesInside[1] = (47.625 + 10) / height;
    index = 2;
    rInside = 47.625 * Math.sqrt(2) - borderThinkness;
    for (i = -40; i <= -10; i++) {
        stoperBottomVerticesInside[index] = rInside * Math.cos(i * 2 * Math.PI / 100) / width;
        stoperBottomVerticesInside[index + 1] = (rInside * Math.sin(i * 2 * Math.PI / 100) + 47.625 + 10) / height;
        index += 2;
    }

    var moldBoldness = 60
    var moldGap = 200
    var moldLeftVertices = new Float32Array([
        -(moldGap + moldBoldness) / width, -150 / height,
        -moldGap / width, -150 / height,
        -(moldGap + moldBoldness) / width, -1600 / height,
        -moldGap / width, -1600 / height
    ])

    var moldLeftVerticesInside = new Float32Array([
        (-(moldGap + moldBoldness) + borderThinkness) / width, (-150 - borderThinkness) / height,
        (-moldGap - borderThinkness) / width, (-150 - borderThinkness) / height,
        (-(moldGap + moldBoldness) + borderThinkness) / width, -1600 / height,
        (-moldGap - borderThinkness) / width, -1600 / height
    ])

    var moldRightVertices = reverseVertices(moldLeftVertices)
    var moldRightVerticesInside = reverseVertices(moldLeftVerticesInside)

    var middleUnknownLeftVertices = new Float32Array([
        -120 / width, 0 / height,
        -36 / width, 0 / height,
        -120 / width, 24 / height,
        -36 / width, 24 / height,
        -85 / width, 59 / height,
        -36 / width, (365 + 45) / height,
        -85 / width, 365 / height,
        -76 / width, 460 / height
    ])
    var middleUnknownLeftVerticesInside = new Float32Array([
        (-120 + borderThinkness) / width, (0 + borderThinkness) / height,
        (-36 - borderThinkness) / width, (0 + borderThinkness) / height,
        (-120 + borderThinkness) / width, (24 - borderThinkness) / height,
        (-36 - borderThinkness) / width, (24 - borderThinkness) / height,
        (-85 + borderThinkness) / width, (59 - borderThinkness) / height,
        (-36 - borderThinkness) / width, (365 + 45 - borderThinkness) / height,
        (-85 + borderThinkness) / width, (365 - borderThinkness) / height,
        (-76 + Math.sqrt((Math.pow(95, 2) + Math.pow(9, 2))) / 95 * borderThinkness) / width, (460 - (borderThinkness)) / height
    ])

    var middleUnknownLeftHeadVertices = new Float32Array(100)
    middleUnknownLeftHeadVertices[0] = -76 / width;
    middleUnknownLeftHeadVertices[1] = 410 / height;
    index = 2;
    r = 40;
    for (i = 0; i <= 50; i++) {
        middleUnknownLeftHeadVertices[index] = (r * Math.cos(i * 2 * Math.PI / 200) - 76) / width;
        middleUnknownLeftHeadVertices[index + 1] = (r * Math.sin(i * 2 * Math.PI / 200) + 410) / height;
        index += 2;
    }

    var middleUnknownLeftHeadVerticesInside = new Float32Array(100)
    middleUnknownLeftHeadVerticesInside[0] = -76 / width;
    middleUnknownLeftHeadVerticesInside[1] = 410 / height;
    index = 2;
    r = 40 - borderThinkness;
    for (i = 0; i <= 50; i++) {
        middleUnknownLeftHeadVerticesInside[index] = (r * Math.cos(i * 2 * Math.PI / 200) - 76) / width;
        middleUnknownLeftHeadVerticesInside[index + 1] = (r * Math.sin(i * 2 * Math.PI / 200) + 410) / height;
        index += 2;
    }

    var middleUnknownRightVertices = reverseVertices(middleUnknownLeftVertices)
    var middleUnknownRightVerticesInside = reverseVertices(middleUnknownLeftVerticesInside)
    var middleUnknownRightHeadVerticesInside = reverseVertices(middleUnknownLeftHeadVerticesInside)
    var middleUnknownRightHeadVertices = reverseVertices(middleUnknownLeftHeadVertices)

    var middleInwardInsideVertice = new Float32Array([
        88 / width, 460 / height,
        -88 / width, 460 / height,
        88 / width, -910 / height,
        -88 / width, -910 / height
    ])

    const dummyBarParams = {
        length : 1000,
        width : 80,
        offset : -400,
        y : -1300
    }

    const dummyBarHeadVertice = new Float32Array([
        (0-dummyBarParams.length - dummyBarParams.offset) / width, dummyBarParams.y/height,
        (0 - dummyBarParams.offset) / width, dummyBarParams.y /height,
        (0-dummyBarParams.offset) / width, (dummyBarParams.y - dummyBarParams.width) / height,
        (0-dummyBarParams.length - dummyBarParams.offset) / width, (dummyBarParams.y-dummyBarParams.width) / height
    ])


    // Define colors
    var tudishColor = new Float32Array([0, 0, 0, 1.0]);
    var stoperColor = new Float32Array([0.8, 0.8, 0.8, 1.0])
    var coolingPipeColor = new Float32Array([0.8, 0.8, 0.8, 1.0]);
    var moldColor = new Float32Array([0.6, 0.6, 0.6, 1.0])
    var borderColor = new Float32Array([0, 0, 0, 1.0])
    var middleInwardColor = new Float32Array([0.5, 0.5, 0.5, 1.0])
    var dummyBarColor = new Float32Array([0.8, 0.3, 0.2, 1.0])

    leftTudish = AnimObjHelper.AnimObj(tudishLeftVertices, tudishColor, gl.TRIANGLE_STRIP, 2);
    leftTudishInside = AnimObjHelper.AnimObj(tudishLeftInsideVertices, new Float32Array([0.8, 0.8, 0.8, 1]), gl.TRIANGLE_STRIP, 2);
    rightTudish = AnimObjHelper.AnimObj(tudishRightVertices, tudishColor, gl.TRIANGLE_STRIP, 2);
    rightTudishInside = AnimObjHelper.AnimObj(tudishRightInsideVertices, new Float32Array([0.8, 0.8, 0.8, 1]), gl.TRIANGLE_STRIP, 2)
    stoper = AnimObjHelper.AnimObj(stoperVertices, borderColor, gl.TRIANGLE_STRIP, 2);
    stoperInside = AnimObjHelper.AnimObj(stoperVerticesInside, stoperColor, gl.TRIANGLE_STRIP, 2);
    stoperBottom = AnimObjHelper.AnimObj(stoperBottomVertices, borderColor, gl.TRIANGLE_FAN, 2);
    stoperBottomInside = AnimObjHelper.AnimObj(stoperBottomVerticesInside, stoperColor, gl.TRIANGLE_FAN, 2);
    leftCoolingPipe = AnimObjHelper.AnimObj(coolingPipeLeftVertices, borderColor, gl.TRIANGLE_FAN, 2);
    rightCoolingPipe = AnimObjHelper.AnimObj(coolingPipeRightVertices, borderColor, gl.TRIANGLE_FAN, 2);
    leftCoolingPipeInside = AnimObjHelper.AnimObj(coolingPipeLeftVerticesInside, coolingPipeColor, gl.TRIANGLE_FAN, 2);
    rightCoolingPipeInside = AnimObjHelper.AnimObj(coolingPipeRightVerticesInside, coolingPipeColor, gl.TRIANGLE_FAN, 2);
    leftCoolingPipeBottom = AnimObjHelper.AnimObj(coolingPipeBottomLeftVertices, borderColor, gl.TRIANGLE_FAN, 2);
    leftCoolingPipeBottomInside = AnimObjHelper.AnimObj(coolingPipeBottomLeftVerticesInside, coolingPipeColor, gl.TRIANGLE_FAN, 2);
    rightCoolingPipeBottom = AnimObjHelper.AnimObj(coolingPipeBottomRightVertices, borderColor, gl.TRIANGLE_FAN, 2);
    rightCoolingPipeBottomInside = AnimObjHelper.AnimObj(coolingPipeBottomRightVerticesInside, coolingPipeColor, gl.TRIANGLE_FAN, 2);
    coolingPipeBreach = AnimObjHelper.AnimObj(coolingPipeBreachVertices, new Float32Array([0, 0, 0, 1]), gl.TRIANGLE_FAN, 2);
    coolingPipeBreachInside = AnimObjHelper.AnimObj(coolingPipeBreachVerticesInside, new Float32Array([1, 1, 1, 1]), gl.TRIANGLE_FAN, 2);

    leftMoldPipe = AnimObjHelper.AnimObj(moldLeftVertices, borderColor, gl.TRIANGLE_STRIP, 2);
    leftMoldPipeInside = AnimObjHelper.AnimObj(moldLeftVerticesInside, moldColor, gl.TRIANGLE_STRIP, 2);
    rightMoldPipe = AnimObjHelper.AnimObj(moldRightVertices, borderColor, gl.TRIANGLE_STRIP, 2);
    rightMoldPipeInside = AnimObjHelper.AnimObj(moldRightVerticesInside, moldColor, gl.TRIANGLE_STRIP, 2);

    // for middle unknown stuffs.
    middleUnknownLeft = AnimObjHelper.AnimObj(middleUnknownLeftVertices, borderColor, gl.TRIANGLE_STRIP, 2);
    middleUnknownLeftInside = AnimObjHelper.AnimObj(middleUnknownLeftVerticesInside, moldColor, gl.TRIANGLE_STRIP, 2);
    middleUnknownLeftHead = AnimObjHelper.AnimObj(middleUnknownLeftHeadVertices, borderColor, gl.TRIANGLE_FAN, 2);
    middleUnknownLeftHeadInside = AnimObjHelper.AnimObj(middleUnknownLeftHeadVerticesInside, moldColor, gl.TRIANGLE_FAN, 2);
    middleUnknownRight = AnimObjHelper.AnimObj(middleUnknownRightVertices, borderColor, gl.TRIANGLE_STRIP, 2);
    middleUnknownRightInside = AnimObjHelper.AnimObj(middleUnknownRightVerticesInside, moldColor, gl.TRIANGLE_STRIP, 2);
    middleUnknownRightHead = AnimObjHelper.AnimObj(middleUnknownRightHeadVertices, borderColor, gl.TRIANGLE_FAN, 2);
    middleUnknownRightHeadInside = AnimObjHelper.AnimObj(middleUnknownRightHeadVerticesInside, moldColor, gl.TRIANGLE_FAN, 2);

    middleInward = AnimObjHelper.AnimObj(middleInwardInsideVertice, middleInwardColor, gl.TRIANGLE_STRIP, 2);
    dummybar = AnimObjHelper.AnimObj(dummyBarHeadVertice, dummyBarColor, gl.TRIANGLE_FAN, 2)

    // Animation object bundles.
    tudishObj = AnimObjHelper.AnimObjBundle([leftTudish, leftTudishInside, rightTudish, rightTudishInside], [0, 59 / height, 0])
    stoperObj = AnimObjHelper.AnimObjBundle([stoper, stoperInside, stoperBottom, stoperBottomInside], [0, 430 / height, 0])
    coolingObj = AnimObjHelper.AnimObjBundle([leftCoolingPipe, rightCoolingPipe, leftCoolingPipeInside, rightCoolingPipeInside, leftCoolingPipeBottom, leftCoolingPipeBottomInside, rightCoolingPipeBottom, rightCoolingPipeBottomInside, coolingPipeBreach, coolingPipeBreachInside])
    moldPipe = AnimObjHelper.AnimObjBundle([leftMoldPipe, leftMoldPipeInside, rightMoldPipe, rightMoldPipeInside])
    middleUnknownObj = AnimObjHelper.AnimObjBundle([middleUnknownLeft, middleUnknownLeftInside, middleUnknownRight, middleUnknownRightInside, middleUnknownLeftHead, middleUnknownLeftHeadInside, middleUnknownRightHead, middleUnknownRightHeadInside])
    middleInwordObj = AnimObjHelper.AnimObjBundle([middleInward])
    dummybarObj = AnimObjHelper.AnimObjBundle([dummybar])
    

    // var animObjs = [leftTudish, rightTudish, leftTudishInside, rightTudishInside, stoper, stoperInside, stoperBottom, stoperBottomInside, leftCoolingPipe,  leftCoolingPipeInside, rightCoolingPipe, rightCoolingPipeInside, leftMoldPipe, leftMoldPipeInside, rightMoldPipe, rightMoldPipeInside,
    //                 middleUnknownLeft, middleUnknownLeftInside, middleUnknownLeftHead, middleUnknownLeftHeadInside, middleUnknownRight, middleUnknownRightInside, middleUnknownRightHead, middleUnknownRightHeadInside];
    var bundles = [middleInwordObj, tudishObj, stoperObj, middleUnknownObj, moldPipe, coolingObj, dummybarObj]

    // Animation
    // var modelMatrix = new Matrix4();
    // modelMatrix.translate(0, 460/height, 0);
    if (AnimationController.session != null) {
        const animSession  = AnimationController.session
        var deltaTime = Date.now() - animSession.startTime;
        deltaNo = parseInt(deltaTime / (250 / 25));
        console.log(deltaNo)
        console.log(animSession.data.stpPos[deltaNo])
        stoperObj.transMatrix = [stoperObj.transMatrix[0], (animSession.data.stpPos[deltaNo] + 460) / height, stoperObj.transMatrix[2]]
        // modelMatrix.translate(0, (translateData.values[deltaNo] + 460) / height, 0);        // Multiply modelMatrix by the calculated translation matrix
    }

    // Init buffer for later use.
    bundles.forEach(bundle => {
        bundle.objects.forEach(animObj => {
            initArrayBufferForLaterUse(gl, animObj.verticeSize, animObj);
        })
    });

    return bundles;
}

function initArrayBufferForLaterUse(gl, num, animObj) {
    var buffer = gl.createBuffer();   // Create a buffer object
    if (!buffer) {
        console.log('Failed to create the buffer object');
        return null;
    }
    // Write date into the buffer object
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, animObj.vertices, gl.STATIC_DRAW);

    // Store the necessary information to assign the object to the attribute variable later
    buffer.num = num;
    buffer.type = gl.FLOAT;

    animObj.buffer = buffer;
}

function drawAnimObjs(gl, a_Position, animObjs) {
    animObjs.forEach(element => {
        drawAnimObj(gl, a_Position, element)
    });
}

function drawAnimObjBundle(gl, a_Position, animObjBundle) {
    animObjBundle.forEach(bundle => {
        bundle.objects.forEach(animObj => {
            var u_Color = gl.getUniformLocation(gl.program, 'u_Color');
            gl.uniform4fv(u_Color, animObj.colors);

            var u_ModelMatrix = gl.getUniformLocation(gl.program, 'u_ModelMatrix');
            if (animObj.modelMatrix != undefined) {
                gl.uniformMatrix4fv(u_ModelMatrix, false, animObj.modelMatrix.elements);
            }
            else {
                var modelMatrix = new Matrix4();
                modelMatrix.translate(bundle.transMatrix[0], bundle.transMatrix[1], bundle.transMatrix[2]);
                gl.uniformMatrix4fv(u_ModelMatrix, false, modelMatrix.elements);
            }

            gl.bindBuffer(gl.ARRAY_BUFFER, animObj.buffer);
            // Assign the buffer object to the attribute variable
            gl.vertexAttribPointer(a_Position, animObj.verticeSize, gl.FLOAT, false, 0, 0);
            // Enable the assignment of the buffer object to the attribute variable
            gl.enableVertexAttribArray(a_Position);

            // Draw
            gl.drawArrays(animObj.drawMethod, 0, animObj.vertices.length / animObj.verticeSize);
        })
    });
}

function drawAnimObj(gl, a_Position, animObj) {

    var u_Color = gl.getUniformLocation(gl.program, 'u_Color');
    gl.uniform4fv(u_Color, animObj.colors);

    var u_ModelMatrix = gl.getUniformLocation(gl.program, 'u_ModelMatrix');
    if (animObj.modelMatrix != undefined) {
        gl.uniformMatrix4fv(u_ModelMatrix, false, animObj.modelMatrix.elements);
    }
    else {
        var modelMatrix = new Matrix4();
        modelMatrix.translate(0, 0, 0);
        gl.uniformMatrix4fv(u_ModelMatrix, false, modelMatrix.elements);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, animObj.buffer);
    // Assign the buffer object to the attribute variable
    gl.vertexAttribPointer(a_Position, animObj.verticeSize, gl.FLOAT, false, 0, 0);
    // Enable the assignment of the buffer object to the attribute variable
    gl.enableVertexAttribArray(a_Position);

    // Draw
    gl.drawArrays(animObj.drawMethod, 0, animObj.vertices.length / animObj.verticeSize);
}

// Just for reference.
function drawAnimObjsOrigin(gl, n, buffer, viewProjMatrix, a_Position, u_MvpMatrix, u_NormalMatrix, animObjs, coordinateSize) {
    gl.bindBuffer(gl.ARRAY_BUFFER, animObjs.buffer);
    // Assign the buffer object to the attribute variable
    gl.vertexAttribPointer(a_Position, coordinateSize, gl.FLOAT, false, 0, 0);
    // Enable the assignment of the buffer object to the attribute variable
    gl.enableVertexAttribArray(a_Position);

    // Calculate the model view project matrix and pass it to u_MvpMatrix
    g_mvpMatrix.set(viewProjMatrix);
    g_mvpMatrix.multiply(g_modelMatrix);
    gl.uniformMatrix4fv(u_MvpMatrix, false, g_mvpMatrix.elements);
    // Calculate matrix for normal and pass it to u_NormalMatrix
    g_normalMatrix.setInverseOf(g_modelMatrix);
    g_normalMatrix.transpose();
    gl.uniformMatrix4fv(u_NormalMatrix, false, g_normalMatrix.elements);

    // Draw
    gl.drawArrays(animObjs.drawMethod, 0, n);
}