function startAnimation() {
    // Init webgl context.
    var canvas = document.getElementById('animationCanvas');
    var canvasDiv = document.getElementById('animationDiv');
    canvas.height = canvasDiv.clientHeight;
    canvas.width = canvasDiv.clientWidth;
    var gl = getWebGLContext(canvas);
    height = canvasDiv.clientHeight;
    width = canvasDiv.clientWidth;
    startDrawing(gl);
}

function startDrawing(gl) {
    gl.clearColor(0, 0, 0, 0.5);
    gl.clear(gl.COLOR_BUFFER_BIT);
    drawTudish(gl);
    drawSTP(gl);
}

function drawSTP(gl) {
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

    var n = initSTPVertexBuffers(gl);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);
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

var height = 2000;
var width = 1000;
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