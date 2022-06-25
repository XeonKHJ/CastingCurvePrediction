const WebGpuWrapper = {
    GLOBAL_VSHADER_SOURCE:
        'attribute vec4 a_Position;\n' +
        'uniform mat4 u_ModelMatrix; \n' +
        'uniform mat4 u_GlobalModelMatrix; \n' +
        'void main() {\n' +
        '  gl_Position = u_GlobalModelMatrix * u_ModelMatrix * a_Position;\n' +
        '}\n',

    // Fragment shader program
    GLOBAL_FSHADER_SOURCE:
        'precision mediump float;\n' +
        'uniform vec4 u_Color;\n' +
        'void main() {\n' +
        '  gl_FragColor = u_Color;\n' +
        '}\n',
}

var AnimationController = {
    isPlaying: true,
    height: 2000,
    width: 1000,
    webGlContext: null,
    session: null,
    renderFrame(gl) {
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
    createSession(data, startTime) {
        return {
            data: data,
            startTime: startTime
        }
    },
    startAnimation(data) {
        this.session = this.createSession(data, Date.now())
    },
    stopAnimation() {
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

function drawAnimation() {
    // Init webgl context.
    var canvas = document.getElementById('animationCanvas');
    var canvasDiv = document.getElementById('animationDiv');
    var gl = getWebGLContext(canvas);
    canvas.height = canvasDiv.clientHeight;
    canvas.width = canvasDiv.clientWidth;

    // Initializev shaders
    if (!initShaders(gl, WebGpuWrapper.GLOBAL_VSHADER_SOURCE, WebGpuWrapper.GLOBAL_FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

    var tick = function () {
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        AnimationController.height = canvasDiv.clientHeight;
        AnimationController.width = canvasDiv.clientWidth;
        AnimationController.renderFrame(gl);
        if (AnimationController.isPlaying) {
            requestAnimationFrame(tick, canvas); // Request that the browser ?calls tick
        }
    };
    tick();
}


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
        borderHeight: 400,
        width: 1500,
        height: 250
    }
    width = 400
    height = 500
    var tudishLeftVertices = new Float32Array(
        [
            -tudishParams.width , (tudishParams.height * tan80) ,
            -(tudishParams.width - ((tudishParams.borderHeight / sin80))) , (tudishParams.height * tan80) ,
            -(tudishParams.width - (tudishParams.height)) , 0,
            (-(tudishParams.width - ((tudishParams.borderHeight / sin80)) - (tudishParams.height * tan80 - tudishParams.borderHeight) / tan80)) , tudishParams.borderHeight ,
            -50 , 0,
            -50 , tudishParams.borderHeight 
        ]
    );

    var tudishRightVertices = reverseVertices(tudishLeftVertices);

    var tudishLeftInsideVertices = new Float32Array(
        [
            (-(tudishParams.width) + borderThinkness / sin80) , ((tudishParams.height * tan80) - borderThinkness) ,
            (-(tudishParams.width - ((tudishParams.borderHeight / sin80))) - borderThinkness / sin80) , ((tudishParams.height * tan80) - borderThinkness) ,
            (-(tudishParams.width - (tudishParams.height)) + borderThinkness / sin80) , (0 + borderThinkness) ,
            ((-(tudishParams.width - ((tudishParams.borderHeight / sin80)) - (tudishParams.height * tan80 - tudishParams.borderHeight) / tan80)) - borderThinkness) , (tudishParams.borderHeight - borderThinkness) ,
            (-50 - borderThinkness) , (0 + borderThinkness) ,
            (-50 - borderThinkness) , (tudishParams.borderHeight - borderThinkness) 
        ]
    );

    var tudishRightInsideVertices = reverseVertices(tudishLeftInsideVertices);

    const stoperParams = {
        // 不包含下面半圆的塞棒长度
        height: 280,
        width: 127,
        // 西方扇形部分的圆半径
        r: 10
    }

    var sin60 = Math.sin(60 / 180 * Math.PI);
    var sin30 = Math.sin(30 / 180 * Math.PI);
    var stoperVertices = new Float32Array([
        -(stoperParams.width / 2) , 300 ,
        (stoperParams.width / 2) , 300 ,
        -(stoperParams.width / 2) , 20 ,
        stoperParams.width / 2 , 20 ,
        -47.625 , 10 ,
        47.625 , 10 
    ]);

    var stoperVerticesInside = new Float32Array([
        (-63.5 + borderThinkness) , (300 - borderThinkness) ,
        (63.5 - borderThinkness) , (300 - borderThinkness) ,
        (-63.5 + borderThinkness) , (20 + borderThinkness) ,
        (63.5 - borderThinkness) , (20 + borderThinkness) ,
        (-47.625 + borderThinkness) , (10 + borderThinkness * sin60 * sin30) ,
        (47.625 - borderThinkness) , (10 + borderThinkness * sin60 * sin30) 
    ]);

    var coolingPipeLeftVertices = new Float32Array([
        -40 , 0 ,
        -120 , 0 ,
        -120 , -44 ,
        -110 , -44 ,
        -90 , -400 ,
        -90 , -700 ,
        -32.5 , -700 
    ]);
    var coolingPipeRightVertices = reverseVertices(coolingPipeLeftVertices);

    var coolingPipeLeftVerticesInside = new Float32Array([
        (-40 - borderThinkness) , (0 - borderThinkness) ,
        (-120 + borderThinkness) , (0 - borderThinkness) ,
        (-120 + borderThinkness) , (-44 + borderThinkness) ,
        (-110 + ((Math.sqrt(Math.pow(356, 2) + Math.pow(10, 2)) - 10) * borderThinkness / 356)) , (-44 + borderThinkness) ,
        (-90 + borderThinkness) , (-400 + 10 / 356 * borderThinkness) ,
        (-90 + borderThinkness) , (-700 + borderThinkness) ,
        (-32.5 - borderThinkness) , (-700 + borderThinkness) 
    ]);

    var coolingPipeRightVerticesInside = reverseVertices(coolingPipeLeftVerticesInside);

    var coolingPipeBottomLeftVertices = new Float32Array([
        -90 , -910 ,
        -90 , -(800 - 100) ,
        -32.5 , -(800 - 100) ,
        -32.5 , -(910 - 100 + 50) ,
        0, -(910 - 100 + 50) ,
        0, -910 
    ])

    var coolingPipeBottomLeftVerticesInside = new Float32Array([
        (-90 + borderThinkness) , (-910 + borderThinkness) ,
        (-90 + borderThinkness) , -(800 - 100 - borderThinkness) ,
        (-32.5 - borderThinkness) , -(800 - 100 - borderThinkness) ,
        (-32.5 - borderThinkness) , -(910 - 100 + 50 + borderThinkness) ,
        0, -(910 - 100 + 50 + borderThinkness) ,
        0, (-910 + borderThinkness) 
    ])

    var coolingPipeBottomRightVertices = reverseVertices(coolingPipeBottomLeftVertices);
    var coolingPipeBottomRightVerticesInside = reverseVertices(coolingPipeBottomLeftVerticesInside)

    var coolingPipeBreachVertices = new Float32Array([
        -32.5 , -700 ,
        32.5 , -700 ,
        32.5 , -810 ,
        -32.5 , -810 
    ])

    var coolingPipeBreachVerticesInside = new Float32Array([
        (-32.5 + borderThinkness) , (-700 - borderThinkness) ,
        (32.5 - borderThinkness) , (-700 - borderThinkness) ,
        (32.5 - borderThinkness) , (-810 + borderThinkness) ,
        (-32.5 + borderThinkness) , (-810 + borderThinkness) 
    ])

    var stoperBottomVertices = new Float32Array(104);
    stoperBottomVertices[0] = 0;
    stoperBottomVertices[1] = (47.625 + 10) ;
    index = 2;
    r = 47.625 * Math.sqrt(2);
    for (i = -40; i <= -10; i++) {
        stoperBottomVertices[index] = r * Math.cos(i * 2 * Math.PI / 100) ;
        stoperBottomVertices[index + 1] = (r * Math.sin(i * 2 * Math.PI / 100) + 47.625 + 10) ;
        index += 2;
    }

    var stoperBottomVerticesInside = new Float32Array(104);
    stoperBottomVerticesInside[0] = 0;
    stoperBottomVerticesInside[1] = (47.625 + 10) ;
    index = 2;
    rInside = 47.625 * Math.sqrt(2) - borderThinkness;
    for (i = -40; i <= -10; i++) {
        stoperBottomVerticesInside[index] = rInside * Math.cos(i * 2 * Math.PI / 100) ;
        stoperBottomVerticesInside[index + 1] = (rInside * Math.sin(i * 2 * Math.PI / 100) + 47.625 + 10) ;
        index += 2;
    }

    var moldBoldness = 60
    var moldGap = 200
    var moldLeftVertices = new Float32Array([
        -(moldGap + moldBoldness) , -150 ,
        -moldGap , -150 ,
        -(moldGap + moldBoldness) , -1600 ,
        -moldGap , -1600 
    ])

    var moldLeftVerticesInside = new Float32Array([
        (-(moldGap + moldBoldness) + borderThinkness) , (-150 - borderThinkness) ,
        (-moldGap - borderThinkness) , (-150 - borderThinkness) ,
        (-(moldGap + moldBoldness) + borderThinkness) , -1600 ,
        (-moldGap - borderThinkness) , -1600 
    ])

    var moldRightVertices = reverseVertices(moldLeftVertices)
    var moldRightVerticesInside = reverseVertices(moldLeftVerticesInside)

    var middleUnknownLeftVertices = new Float32Array([
        -120 , 0 ,
        -36 , 0 ,
        -120 , 24 ,
        -36 , 24 ,
        -85 , 59 ,
        -36 , (365 + 45) ,
        -85 , 365 ,
        -76 , 460 
    ])
    var middleUnknownLeftVerticesInside = new Float32Array([
        (-120 + borderThinkness) , (0 + borderThinkness) ,
        (-36 - borderThinkness) , (0 + borderThinkness) ,
        (-120 + borderThinkness) , (24 - borderThinkness) ,
        (-36 - borderThinkness) , (24 - borderThinkness) ,
        (-85 + borderThinkness) , (59 - borderThinkness) ,
        (-36 - borderThinkness) , (365 + 45 - borderThinkness) ,
        (-85 + borderThinkness) , (365 - borderThinkness) ,
        (-76 + Math.sqrt((Math.pow(95, 2) + Math.pow(9, 2))) / 95 * borderThinkness) , (460 - (borderThinkness)) 
    ])

    var middleUnknownLeftHeadVertices = new Float32Array(100)
    middleUnknownLeftHeadVertices[0] = -76 ;
    middleUnknownLeftHeadVertices[1] = 410 ;
    index = 2;
    r = 40;
    for (i = 0; i <= 50; i++) {
        middleUnknownLeftHeadVertices[index] = (r * Math.cos(i * 2 * Math.PI / 200) - 76) ;
        middleUnknownLeftHeadVertices[index + 1] = (r * Math.sin(i * 2 * Math.PI / 200) + 410) ;
        index += 2;
    }

    var middleUnknownLeftHeadVerticesInside = new Float32Array(100)
    middleUnknownLeftHeadVerticesInside[0] = -76 ;
    middleUnknownLeftHeadVerticesInside[1] = 410 ;
    index = 2;
    r = 40 - borderThinkness;
    for (i = 0; i <= 50; i++) {
        middleUnknownLeftHeadVerticesInside[index] = (r * Math.cos(i * 2 * Math.PI / 200) - 76) ;
        middleUnknownLeftHeadVerticesInside[index + 1] = (r * Math.sin(i * 2 * Math.PI / 200) + 410) ;
        index += 2;
    }

    var middleUnknownRightVertices = reverseVertices(middleUnknownLeftVertices)
    var middleUnknownRightVerticesInside = reverseVertices(middleUnknownLeftVerticesInside)
    var middleUnknownRightHeadVerticesInside = reverseVertices(middleUnknownLeftHeadVerticesInside)
    var middleUnknownRightHeadVertices = reverseVertices(middleUnknownLeftHeadVertices)

    var middleInwardInsideVertice = new Float32Array([
        88 , 460 ,
        -88 , 460 ,
        88 , -910 ,
        -88 , -910 
    ])

    const dummyBarParams = {
        length: 1000,
        width: 80,
        offset: -400,
        y: -1300
    }

    const dummyBarHeadVertice = new Float32Array([
        (0 - dummyBarParams.length - dummyBarParams.offset) , dummyBarParams.y ,
        (0 - dummyBarParams.offset) , dummyBarParams.y ,
        (0 - dummyBarParams.offset) , (dummyBarParams.y - dummyBarParams.width) ,
        (0 - dummyBarParams.length - dummyBarParams.offset) , (dummyBarParams.y - dummyBarParams.width) 
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
    tudishObj = AnimObjHelper.AnimObjBundle([leftTudish, leftTudishInside, rightTudish, rightTudishInside], [0, 59, 0])
    stoperObj = AnimObjHelper.AnimObjBundle([stoper, stoperInside, stoperBottom, stoperBottomInside], [0, 430, 0])
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
        const animSession = AnimationController.session
        var deltaTime = Date.now() - animSession.startTime;
        deltaNo = parseInt(deltaTime / (250 / 25));
        console.log(deltaNo)
        console.log(animSession.data.stpPos[deltaNo])
        stoperObj.transMatrix = [stoperObj.transMatrix[0], (animSession.data.stpPos[deltaNo] + 460), stoperObj.transMatrix[2]]
        // modelMatrix.translate(0, (translateData.values[deltaNo] + 460) / height, 0);        // Multiply modelMatrix by the calculated translation matrix
    }

    // Init buffer for later use.
    bundles.forEach(bundle => {
        bundle.objects.forEach(animObj => {
            initArrayBufferForLaterUse(gl, animObj.verticeSize, animObj);
        })
        for(var i = 0; i < bundle.transMatrix.length; ++i)
        {
            bundle.transMatrix[i] = bundle.transMatrix[i] /  ((i % 2 == 0) ? AnimationController.width : AnimationController.height)
        }
    });

    return bundles;
}

function initArrayBufferForLaterUse(gl, num, animObj) {
    for(var i = 0; i < animObj.vertices.length; ++i)
    {
        animObj.vertices[i] = animObj.vertices[i] / ((i % 2 == 0) ? AnimationController.width : AnimationController.height)
    }

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