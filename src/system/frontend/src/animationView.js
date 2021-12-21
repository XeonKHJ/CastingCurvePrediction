var _animationStarted = true;

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

    // Initialize shaders
    if (!initShaders(gl, GLOBAL_VSHADER_SOURCE, GLOBAL_FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

    var tick = function () {
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        height = canvasDiv.clientHeight;
        width = canvasDiv.clientWidth;
        startDrawing(gl);
        if (_animationStarted) {
            requestAnimationFrame(tick, canvas); // Request that the browser ?calls tick
        }
    };
    tick();
}

function startDrawing(gl) {
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    animObjs = initVertices(gl);

    // set global scale matrix
    var u_GlobalModelMatrix = gl.getUniformLocation(gl.program, 'u_GlobalModelMatrix');
    var globalModelMatrix = new Matrix4();
    globalModelMatrix.translate(0, 0.7, 0);
    globalModelMatrix.scale(0.5,0.5, 1)
    gl.uniformMatrix4fv(u_GlobalModelMatrix, false, globalModelMatrix.elements)


    var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
    // var u_MvpMatrix = gl.getUniformLocation(gl.program, 'u_MvpMatrix');
    // var u_NormalMatrix = gl.getUniformLocation(gl.program, 'u_NormalMatrix');

    drawAnimObjs(gl, a_Position, animObjs);
}

var startTime = null;
var translateData = null;

var height = 2000;
var width = 1000;
var offset = 0;

function AnimObj(vertices, colors, drawMethod, verticeSize) {
    return {
        vertices: vertices,
        drawMethod: drawMethod,
        verticeSize: verticeSize,
        colors: colors,
        buffer: null
    }
}

function initVertices(gl) {
    var tan = Math.tan(80 / 180 * Math.PI);
    var tudishLeftVertices = new Float32Array(
        [
            -400 / width, (50 * tan) / height,
            -350 / width, (50 * tan) / height,
            -350 / width, 0,
            (-350 + ((50 * tan - 50) / tan)) / width, 50 / height,
            -50 / width, 0,
            -50 / width, 50 / height
        ]
    );

    var tudishRightVertices = new Float32Array(
        12
    );
    for (var i = 0; i < tudishLeftVertices.length; i++) {
        if (i % 2) {
            tudishRightVertices[i] = tudishLeftVertices[i];
        }
        else {
            tudishRightVertices[i] = -1 * tudishLeftVertices[i];
        }
    }

    var stoperVertices = new Float32Array([
        -63.5 / width, 300 / height,
        63.5 / width, 300 / height,
        -63.5 / width, 40 / height,
        63.5 / width, 40 / height,
        -40 / width, 0,
        40 / width, 0
    ]);

    var coolingPipeLeftVertices = new Float32Array([
        -40 / width, 0 / height,
        -120 / width, 0 / height,
        -120 / width, -44 / height,
        -110 / width, -44 / height,
        -90 / width, -400 / height,
        -90 / width, -800 / height,
        -32.5 / width, -800 / height
    ]);
    var coolingPipeRightVertices = new Float32Array(coolingPipeLeftVertices.length);
    for (var i = 0; i < coolingPipeLeftVertices.length; i++) {
        if (i % 2) {
            coolingPipeRightVertices[i] = coolingPipeLeftVertices[i];
        }
        else {
            coolingPipeRightVertices[i] = -1 * coolingPipeLeftVertices[i];
        }
    }

    var stoperBottomVertices = new Float32Array(104);
    stoperBottomVertices[0] = 0;
    stoperBottomVertices[1] = 0;
    index = 2;
    const r = 40;
    for (i = -50; i <= 0; i++) {
        stoperBottomVertices[index] = r * Math.cos(i * 2 * Math.PI / 100) / width;
        stoperBottomVertices[index + 1] = r * Math.sin(i * 2 * Math.PI / 100) / height;
        index += 2;
    }
    
    var moldLeftVertices = new Float32Array([
        -180/width, -300/height,
        -150/width, -300/height,
        -180/width, -2000/height,
        -150/width, -2000/height
    ])
    var moldRightVertices = new Float32Array([
        180/width, -300/height,
        150/width, -300/height,
        180/width, -2000/height,
        150/width, -2000/height
    ])

    // Define colors
    var tudishColor = new Float32Array([0.8, 0.8, 0.8, 1.0]);
    var stoperColor = new Float32Array([0.8, 0.8, 0.8, 1.0])
    var coolingPipeColor = new Float32Array([0.8, 0.8, 0.8, 1.0]);
    var moldColor = new Float32Array([0, 0, 0, 1.0])

    leftTudish = AnimObj(tudishLeftVertices, tudishColor, gl.TRIANGLE_STRIP, 2);
    rightTudish = AnimObj(tudishRightVertices, tudishColor, gl.TRIANGLE_STRIP, 2);
    stoper = AnimObj(stoperVertices, stoperColor, gl.TRIANGLE_STRIP, 2);
    stoperBottom = AnimObj(stoperBottomVertices, stoperColor, gl.TRIANGLE_FAN, 2);
    leftCoolingPipe = AnimObj(coolingPipeLeftVertices, coolingPipeColor, gl.TRIANGLE_FAN, 2);
    rightCoolingPipe = AnimObj(coolingPipeRightVertices, coolingPipeColor, gl.TRIANGLE_FAN, 2);
    leftMoldPipe = AnimObj(moldLeftVertices, moldColor, gl.TRIANGLE_STRIP, 2);
    rightMoldPipe = AnimObj(moldRightVertices, moldColor, gl.TRIANGLE_STRIP, 2);

    var animObjs = [leftTudish, rightTudish, stoper, stoperBottom, leftCoolingPipe, rightCoolingPipe, leftMoldPipe, rightMoldPipe];

    // Animation
    var modelMatrix = new Matrix4();
    if (startTime != null) {
        var deltaTime = Date.now() - startTime;
        deltaNo = parseInt(deltaTime / (250 / 25));
        console.log(deltaNo)
        console.log(translateData.values[deltaNo])
        modelMatrix.translate(0, translateData.values[deltaNo] / height, 0);        // Multiply modelMatrix by the calculated translation matrix
    }
    stoper.modelMatrix = modelMatrix;
    stoperBottom.modelMatrix = modelMatrix;

    // Init buffer for later use.
    animObjs.forEach(animObj => {
        initArrayBufferForLaterUse(gl, animObj.verticeSize, animObj);
    });

    return animObjs;
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