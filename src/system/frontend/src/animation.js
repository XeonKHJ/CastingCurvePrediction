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

function main() {
    var canvas = document.getElementById('webgl');

    var gl = getWebGLContext(canvas);
    if (!gl) {
        console.log('Failed to get the rendering context.');
        return;
    }

    if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
        console.log('Failed to intialize shaders.');
        return;
    }

    var n = initVertexBuffersNew(gl);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }

    gl.clearColor(0, 0, 0, 0.5);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);

    n = initVertexBuffersNew2(gl);
    if (n < 0) {
        console.log('Failed to set the positions of the vertices');
        return;
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);
}

function initVertexBuffers(gl) {
    var vertices = new Float32Array([
      -0.5, 0.5,   -0.5, -0.5,   0.5, 0.5,　0.5, -0.5
    ]);
    var n = 4; // The number of vertices
  
    // Create a buffer object
    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
      console.log('Failed to create the buffer object');
      return -1;
    }
  
    // Bind the buffer object to target
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    // Write date into the buffer object
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
    var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
    if (a_Position < 0) {
      console.log('Failed to get the storage location of a_Position');
      return -1;
    }
  　// Assign the buffer object to a_Position variable
    gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, 0, 0);
  
    // Enable the assignment to a_Position variable
    gl.enableVertexAttribArray(a_Position);
  
    return n;
  }

function initVertexBuffersNew(gl) {
    var height = 2000;
    var width = 1000;
    var tudishAngle = 80;
    var tan = Math.tan(80/180 * Math.PI);
    var tudishLeftVertices = new Float32Array(
        [
            -400/width, (50 * tan)/height, 
            -350/width, (50 * tan)/height, 
            -350/width, 0,
            (-350 + ((50*tan - 50) / tan)) /width, 50/height,
            -50/width, 0,
            -50/width, 50/height
        ]
    );
    var tudishRightVertices = new Float32Array(
        12
    );

    for(var i = 0; i < tudishLeftVertices.length; i++)
    {
        if(i % 2)
        {
            tudishRightVertices[i] = tudishLeftVertices[i];
        }
        else
        {
            tudishRightVertices[i] = -1 * tudishLeftVertices[i];
        }
    }


    // var tudishVertices = new Float32Array([
    //     -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -0.5
    // ]);

    var n = tudishLeftVertices.length / 2;
    var stride = tudishLeftVertices.BYTES_PER_ELEMENT;

    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
        console.log('Failed to create the buffer object');
        return -1;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, tudishLeftVertices, gl.STATIC_DRAW)

    var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
    if (a_Position < 0) {
        console.log('Filed to get the strorage location of a_Position')
        return -1;
    }
    gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_Position)
    
    return n;
}

function initVertexBuffersNew2(gl) {
    var height = 2000;
    var width = 1000;
    var tudishAngle = 80;
    var tan = Math.tan(80/180 * Math.PI);
    var tudishLeftVertices = new Float32Array(
        [
            -400/width, (50 * tan)/height, 
            -350/width, (50 * tan)/height, 
            -350/width, 0,
            (-350 + ((50*tan - 50) / tan)) /width, 50/height,
            -50/width, 0,
            -50/width, 50/height
        ]
    );
    var tudishRightVertices = new Float32Array(
        12
    );

    for(var i = 0; i < tudishLeftVertices.length; i++)
    {
        if(i % 2)
        {
            tudishRightVertices[i] = tudishLeftVertices[i];
        }
        else
        {
            tudishRightVertices[i] = -1 * tudishLeftVertices[i];
        }
    }


    // var tudishVertices = new Float32Array([
    //     -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -0.5
    // ]);

    var n = tudishLeftVertices.length / 2;
    var stride = tudishLeftVertices.BYTES_PER_ELEMENT;

    var vertexBuffer = gl.createBuffer();
    if (!vertexBuffer) {
        console.log('Failed to create the buffer object');
        return -1;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, tudishRightVertices, gl.STATIC_DRAW)

    var a_Position = gl.getAttribLocation(gl.program, 'a_Position');
    if (a_Position < 0) {
        console.log('Filed to get the strorage location of a_Position')
        return -1;
    }
    gl.vertexAttribPointer(a_Position, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(a_Position)
    
    return n;
}

function readFileAsString(path) {
    var files = this.files;
    if (files.length === 0) {
        console.log('No file is selected');
        return;
    }

    var reader = new FileReader();
    reader.onload = function (event) {
        console.log('File content:', event.target.result);
    };
    reader.readAsText(files[0]);
}