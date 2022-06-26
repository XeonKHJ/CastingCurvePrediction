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

// Build an animation object requires following steps:
// 1. Create vertice arrays.
// 2. Create AnimObj from vertice arrays.
// 3. Create AnimObjBundle that includes all AnimObjs for this object.

const animConfig = {
    borderThinkness: 3,
    defaultColor: new Float32Array([0, 0, 0, 1.0]),
    defaultBorderColor: new Float32Array([0, 0, 0, 1.0])
}

const mathMiscs = {
    sin60: Math.sin(60 / 180 * Math.PI),
    sin30: Math.sin(30 / 180 * Math.PI)
}

const stoperParams = {
    // 颜色
    color: new Float32Array([0.8, 0.8, 0.8, 1.0]),
    // 不包含下面半圆的塞棒长度
    height: 280,
    width: 127,
    // 西方扇形部分的圆半径
    r: 10
}

const stoperAnimBundleBuilder = {
    /**
    * 该建造器的初始化函数，使用该构造器时要用改构造函数返回的实例。
    * @param offset {Float} 塞棒移动的距离
    * @return 塞棒动画结构的建造器
    */
    init(offset) {
        this.offset = offset
        return this
    },
    buildVertices() {
        const sin60 = Math.sin(60 / 180 * Math.PI);
        const sin30 = Math.sin(30 / 180 * Math.PI);
        this.stoperVertices = new Float32Array([
            -(stoperParams.width / 2), 300,
            (stoperParams.width / 2), 300,
            -(stoperParams.width / 2), 20,
            stoperParams.width / 2, 20,
            -47.625, 10,
            47.625, 10
        ]);
        this.stoperVerticesInside = new Float32Array([
            (-63.5 + animConfig.borderThinkness), (300 - animConfig.borderThinkness),
            (63.5 - animConfig.borderThinkness), (300 - animConfig.borderThinkness),
            (-63.5 + animConfig.borderThinkness), (20 + animConfig.borderThinkness),
            (63.5 - animConfig.borderThinkness), (20 + animConfig.borderThinkness),
            (-47.625 + animConfig.borderThinkness), (10 + animConfig.borderThinkness * sin60 * sin30),
            (47.625 - animConfig.borderThinkness), (10 + animConfig.borderThinkness * sin60 * sin30)
        ]);

        this.stoperBottomVertices = new Float32Array(104);
        this.stoperBottomVertices[0] = 0;
        this.stoperBottomVertices[1] = (47.625 + 10);
        let index = 2;
        let r = 47.625 * Math.sqrt(2);
        for (i = -40; i <= -10; i++) {
            this.stoperBottomVertices[index] = r * Math.cos(i * 2 * Math.PI / 100);
            this.stoperBottomVertices[index + 1] = (r * Math.sin(i * 2 * Math.PI / 100) + 47.625 + 10);
            index += 2;
        }

        this.stoperBottomVerticesInside = new Float32Array(104);
        this.stoperBottomVerticesInside[0] = 0;
        this.stoperBottomVerticesInside[1] = (47.625 + 10);
        index = 2;
        let rInside = 47.625 * Math.sqrt(2) - animConfig.borderThinkness;
        for (i = -40; i <= -10; i++) {
            this.stoperBottomVerticesInside[index] = rInside * Math.cos(i * 2 * Math.PI / 100);
            this.stoperBottomVerticesInside[index + 1] = (rInside * Math.sin(i * 2 * Math.PI / 100) + 47.625 + 10);
            index += 2;
        }
    },

    buildAnimObj(gl) {
        this.stoperObj = AnimObjHelper.AnimObj(this.stoperVertices, animConfig.defaultBorderColor, gl.TRIANGLE_STRIP, 2);
        this.stoperInsideObj = AnimObjHelper.AnimObj(this.stoperVerticesInside, stoperParams.color, gl.TRIANGLE_STRIP, 2);
        this.stoperBottom = AnimObjHelper.AnimObj(this.stoperBottomVertices, animConfig.defaultBorderColor, gl.TRIANGLE_FAN, 2);
        this.stoperBottomInside = AnimObjHelper.AnimObj(this.stoperBottomVerticesInside, stoperParams.color, gl.TRIANGLE_FAN, 2);
    },

    buildBundle() {
        this.stoperBundle = AnimObjHelper.AnimObjBundle([this.stoperObj, this.stoperInsideObj, this.stoperBottom, this.stoperBottomInside], [0, 430, 0])
        this.stoperBundle.transMatrix = [this.stoperBundle.transMatrix[0], (this.offset + 460), this.stoperBundle.transMatrix[2]]
    },
    build(gl) {
        this.buildVertices();
        this.buildAnimObj(gl);
        this.buildBundle();
        return this.stoperBundle;
    }
}

