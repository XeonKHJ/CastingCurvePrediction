var _animationStarted = false;

function drawAnimation() {
    // Init webgl context.
    var canvas = document.getElementById('animationCanvas');
    var canvasDiv = document.getElementById('animationDiv');
    var gl = getWebGLContext(canvas);
    canvas.height = canvasDiv.clientHeight;
    canvas.width = canvasDiv.clientWidth;
    var tick = function () {
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        height = canvasDiv.clientHeight;
        width = canvasDiv.clientWidth;
        startDrawing(gl);
        requestAnimationFrame(tick, canvas); // Request that the browser ?calls tick
    };
    tick();
}

function startDrawing(gl) {
    gl.clearColor(0, 0, 0, 0.5);
    gl.clear(gl.COLOR_BUFFER_BIT);
    drawTudish(gl);
    drawNewSTP(gl);
    drawCoolingPipe(gl);
}

function drawSTP(gl) {
    var VSHADER_SOURCE =
        'attribute vec4 a_Position;\n' +
        'uniform mat4 u_ModelMatrix; \n' +
        'void main() {\n' +
        '  gl_Position = u_ModelMatrix * a_Position;\n' +
        '}\n';

    // Fragment shader program
    var FSHADER_SOURCE =
        'void main() {\n' +
        '  gl_FragColor = vec4(0.0, 1.0, 1.0, 1.0);\n' +
        '}\n';

    if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

    var n = initSTPVertexBuffers(gl);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);
}

var startTime = null;
var translateData = null;

function drawNewSTP(gl) {
    var VSHADER_SOURCE =
        'attribute vec4 a_Position;\n' +
        'uniform mat4 u_ModelMatrix; \n' +
        'void main() {\n' +
        '  gl_Position = u_ModelMatrix * a_Position;\n' +
        '}\n';

    // Fragment shader program
    var FSHADER_SOURCE =
        'void main() {\n' +
        '  gl_FragColor = vec4(0.0, 1.0, 1.0, 1.0);\n' +
        '}\n';

    if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

    var modelMatrix = new Matrix4();

    if(startTime != null)
    {
        var deltaTime = Date.now() - startTime;
        deltaNo = parseInt(deltaTime / (250 / 25)); 
        console.log(deltaNo)
        console.log(translateData.values[deltaNo])
        modelMatrix.translate(0, translateData.values[deltaNo] / height, 0);        // Multiply modelMatrix by the calculated translation matrix
    }

    var u_ModelMatrix = gl.getUniformLocation(gl.program, 'u_ModelMatrix');
    gl.uniformMatrix4fv(u_ModelMatrix, false, modelMatrix.elements);

    var n = initNewSTPVertexBuffers(gl);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);


    var n = initSTPBottomVerticeBuffer(gl, 40);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }
    gl.drawArrays(gl.TRIANGLE_FAN, 0, n)
}

function initSTPVertexBuffers(gl) {
    var vertices = new Float32Array([
        -50 / width, 500 / height,
        50 / width, 500 / height,
        -50 / width, 50 / height,
        50 / width, 50 / height,
        0, 0
    ]);

    var n = vertices.length / 2;

    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
        console.log('Failed to create the buffer object');
        return -1;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
    if (a_Position < 0) {
        console.log('Filed to get the strorage location of a_Position')
        return -1;
    }
    gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_Position)

    return n;
}

function initNewSTPVertexBuffers(gl) {
    var vertices = new Float32Array([
        -63.5 / width, 300 / height,
        63.5 / width, 300 / height,
        -63.5 / width, 40 / height,
        63.5 / width, 40 / height,
        -40 / width, 0,
        40 / width, 0
    ]);

    var n = vertices.length / 2;

    if (!initArrayBuffer(gl, 'a_Position', vertices, 2, gl.FLOAT)) {
        return -1;
    }

    return n;
}

function initSTPBottomVerticeBuffer(gl, r) {
    var vertices = new Float32Array(104);
    vertices[0] = 0;
    vertices[1] = 0;
    index = 2;
    for (i = -50; i <= 0; i++) {
        vertices[index] = r * Math.cos(i * 2 * Math.PI / 100) / width;
        vertices[index + 1] = r * Math.sin(i * 2 * Math.PI / 100) / height;
        index += 2;
    }

    var n = vertices.length / 2;

    if (!initArrayBuffer(gl, 'a_Position', vertices, 2, gl.FLOAT)) {
        return -1;
    }

    return n;
}

var height = 2000;
var width = 1000;
var offset = 0;
function drawTudish(gl) {
    var VSHADER_SOURCE =
        'attribute vec4 a_Position;\n' +
        'void main() {\n' +
        '  gl_Position = a_Position;\n' +
        '}\n';

    // Fragment shader program
    var FSHADER_SOURCE =
        'void main() {\n' +
        '  gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);\n' +
        '}\n';

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

    if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

    var n = initTudishVertexBuffers(gl, tudishLeftVertices);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);

    n = initTudishVertexBuffers(gl, tudishRightVertices);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);
}

function initTudishVertexBuffers(gl, vertices) {
    var n = vertices.length / 2;

    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
        console.log('Failed to create the buffer object');
        return -1;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
    if (a_Position < 0) {
        console.log('Filed to get the strorage location of a_Position')
        return -1;
    }
    gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_Position)

    return n;
}


function initCoolingPipeBuffers(gl, vertices) {
    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
        console.log('Failed to create the buffer object');
        return -1;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
    if (a_Position < 0) {
        console.log('Filed to get the strorage location of a_Position')
        return -1;
    }
    gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_Position)

    return vertices.length / 2;
}

function drawCoolingPipe(gl) {
    var VSHADER_SOURCE =
        'attribute vec4 a_Position;\n' +
        'uniform mat4 u_ModelMatrix; \n' +
        'void main() {\n' +
        '  gl_Position = a_Position;\n' +
        '}\n';

    // Fragment shader program
    var FSHADER_SOURCE =
        'void main() {\n' +
        '  gl_FragColor = vec4(0.0, 1.0, 1.0, 1.0);\n' +
        '}\n';

    if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

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

    // Initialize gl and start drawing

    var n = initCoolingPipeBuffers(gl, coolingPipeLeftVertices);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }
    gl.drawArrays(gl.TRIANGLE_FAN, 0, n);


    var n = initCoolingPipeBuffers(gl, coolingPipeRightVertices);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }
    gl.drawArrays(gl.TRIANGLE_FAN, 0, n);
}

function initArrayBuffer(gl, attribute, data, num, type) {
    // Create a buffer object
    var buffer = gl.createBuffer();
    if (!buffer) {
        console.log('Failed to create the buffer object');
        return false;
    }
    // Write date into the buffer object
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    // Assign the buffer object to the attribute variable
    var a_attribute = gl.getAttribLocation(gl.program, attribute);
    if (a_attribute < 0) {
        console.log('Failed to get the storage location of ' + attribute);
        return false;
    }
    gl.vertexAttribPointer(a_attribute, num, type, false, 0, 0);
    // Enable the assignment of the buffer object to the attribute variable
    gl.enableVertexAttribArray(a_attribute);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    return true;
}