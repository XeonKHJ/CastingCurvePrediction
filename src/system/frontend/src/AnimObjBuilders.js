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

const verticeUtils = {
    reverse(verticesIn) {
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

const tudishParams = {
    borderHeight: 260,
    width: 1085,
    height: 332.65, // 1565 / tan78得到的。
    color: [0, 0, 0, 1.0]
}
const tudishAnimBundleBuilder = {
    /**
    * 该建造器的初始化函数，使用该构造器时要用改构造函数返回的实例。
    */
    init() {
        return this
    },
    buildVertices() {
        let tan80 = Math.tan(a2r(78));
        let sin80 = Math.sin(a2r(78));
        this.tudishLeftVertices = new Float32Array(
            [
                -tudishParams.width, (tudishParams.height * tan80),
                -(tudishParams.width - ((tudishParams.borderHeight / sin80))), (tudishParams.height * tan80),
                -(tudishParams.width - (tudishParams.height)), 0,
                (-(tudishParams.width - ((tudishParams.borderHeight / sin80)) - (tudishParams.height * tan80 - tudishParams.borderHeight) / tan80)), tudishParams.borderHeight,
                -50, 0,
                -50, tudishParams.borderHeight
            ]
        );
        this.tudishRightVertices = verticeUtils.reverse(this.tudishLeftVertices);

        this.tudishLeftInsideVertices = new Float32Array(
            [
                (-(tudishParams.width) + animConfig.borderThinkness / sin80), ((tudishParams.height * tan80) - animConfig.borderThinkness),
                (-(tudishParams.width - ((tudishParams.borderHeight / sin80))) - animConfig.borderThinkness / sin80), ((tudishParams.height * tan80) - animConfig.borderThinkness),
                (-(tudishParams.width - (tudishParams.height)) + animConfig.borderThinkness / sin80), (0 + animConfig.borderThinkness),
                ((-(tudishParams.width - ((tudishParams.borderHeight / sin80)) - (tudishParams.height * tan80 - tudishParams.borderHeight) / tan80)) - animConfig.borderThinkness), (tudishParams.borderHeight - animConfig.borderThinkness),
                (-50 - animConfig.borderThinkness), (0 + animConfig.borderThinkness),
                (-50 - animConfig.borderThinkness), (tudishParams.borderHeight - animConfig.borderThinkness)
            ]
        );

        this.tudishRightInsideVertices = verticeUtils.reverse(this.tudishLeftInsideVertices);
    },
    buildAnimObj(gl) {
        this.leftTudish = AnimObjHelper.AnimObj(this.tudishLeftVertices, tudishParams.color, gl.TRIANGLE_STRIP, 2);
        this.leftTudishInside = AnimObjHelper.AnimObj(this.tudishLeftInsideVertices, new Float32Array([0.8, 0.8, 0.8, 1]), gl.TRIANGLE_STRIP, 2);
        this.rightTudish = AnimObjHelper.AnimObj(this.tudishRightVertices, tudishParams.color, gl.TRIANGLE_STRIP, 2);
        this.rightTudishInside = AnimObjHelper.AnimObj(this.tudishRightInsideVertices, new Float32Array([0.8, 0.8, 0.8, 1]), gl.TRIANGLE_STRIP, 2)
    },
    buildBundle() {
        this.bundle = AnimObjHelper.AnimObjBundle([this.leftTudish, this.leftTudishInside, this.rightTudish, this.rightTudishInside], [0, 59, 0])
    },
    build(gl) {
        this.buildVertices();
        this.buildAnimObj(gl);
        this.buildBundle();
        return this.bundle;
    }
}

const moldParams = {
    color: new Float32Array([0.6, 0.6, 0.6, 1.0]),
    moldBoldness: 60,
    gap: 750,
}
const moldAnimBundleBuilder = {
    init() {
        return this
    },
    buildVertices() {
        this.moldLeftVertices = new Float32Array([
            -(moldParams.gap + moldParams.moldBoldness), -150,
            -moldParams.gap, -150,
            -(moldParams.gap + moldParams.moldBoldness), -1600,
            -moldParams.gap, -1600
        ])

        this.moldLeftVerticesInside = new Float32Array([
            (-(moldParams.gap + moldParams.moldBoldness) + animConfig.borderThinkness), (-150 - animConfig.borderThinkness),
            (-moldParams.gap - animConfig.borderThinkness), (-150 - animConfig.borderThinkness),
            (-(moldParams.gap + moldParams.moldBoldness) + animConfig.borderThinkness), -1600,
            (-moldParams.gap - animConfig.borderThinkness), -1600
        ])

        this.moldRightVertices = verticeUtils.reverse(this.moldLeftVertices)
        this.moldRightVerticesInside = verticeUtils.reverse(this.moldLeftVerticesInside)
    },
    buildAnimObj(gl) {
        this.leftMoldPipe = AnimObjHelper.AnimObj(this.moldLeftVertices, animConfig.defaultBorderColor, gl.TRIANGLE_STRIP, 2);
        this.leftMoldPipeInside = AnimObjHelper.AnimObj(this.moldLeftVerticesInside, moldParams.color, gl.TRIANGLE_STRIP, 2);
        this.rightMoldPipe = AnimObjHelper.AnimObj(this.moldRightVertices, animConfig.defaultBorderColor, gl.TRIANGLE_STRIP, 2);
        this.rightMoldPipeInside = AnimObjHelper.AnimObj(this.moldRightVerticesInside, moldParams.color, gl.TRIANGLE_STRIP, 2);
    },
    buildBundle() {
        this.bundle = moldPipe = AnimObjHelper.AnimObjBundle([this.leftMoldPipe, this.leftMoldPipeInside, this.rightMoldPipe, this.rightMoldPipeInside])
    },
    build(gl) {
        this.buildVertices();
        this.buildAnimObj(gl);
        this.buildBundle();
        return this.bundle;
    }
}

const steelLiquidParams = {
    //颜色
    color: new Float32Array([0.8, 0.3, 0.2, 1.0]),
    // 中包中的液体高度。从下到上计算
    tudishHeight: tudishParams.height,
    // 冷凝管中的液体高度。从上到下开始算
    coolingPipe: 0,
}
const steelLiquidAnimBundleBuilder = {
    init(height, liquidInModeHeight) {
        this.liquidHeight = height
        steelLiquidParams.tudishHeight = height
        this.liquidInMoldHeight = liquidInModeHeight
        return this;
    },
    buildVertices() {
        let sin80 = Math.sin(a2r(70));
        this.steelLiquidInTudishVertices = new Float32Array([
            -(tudishParams.width - ((tudishParams.borderHeight / sin80))), 50 + steelLiquidParams.tudishHeight + tudishParams.height,
            (tudishParams.width - ((tudishParams.borderHeight / sin80))), 50 + steelLiquidParams.tudishHeight + tudishParams.height,
            -(tudishParams.width - ((tudishParams.borderHeight / sin80))), 50 + tudishParams.height,
            (tudishParams.width - ((tudishParams.borderHeight / sin80))), 50 + tudishParams.height
        ])

        this.steelLiquidInMoldVertices = new Float32Array([
            -moldParams.gap, -1300,
            -moldParams.gap, -(1300 - this.liquidInMoldHeight),
            moldParams.gap, -1300,
            moldParams.gap, -(1300 - this.liquidInMoldHeight)
        ])
    },
    buildAnimObj(gl) {
        this.steelLiquidInTudish = AnimObjHelper.AnimObj(this.steelLiquidInTudishVertices, steelLiquidParams.color, gl.TRIANGLE_STRIP, 2)
        this.steelLiquidInMold = AnimObjHelper.AnimObj(this.steelLiquidInMoldVertices, steelLiquidParams.color, gl.TRIANGLE_STRIP, 2)
    },
    buildBundle() {
        this.bundle = AnimObjHelper.AnimObjBundle([this.steelLiquidInTudish, this.steelLiquidInMold])
    },
    build(gl) {
        this.buildVertices();
        this.buildAnimObj(gl);
        this.buildBundle();
        return this.bundle
    }
}

const ladleParams = {
    borderHeight: 100,
    width: 1500,
    height: 200,
    color: [0, 0, 0, 1.0],
    gap: 200,
    gapPos: 20, // start from left
}
const ladleBuilder = {
    init() {
        return this
    },
    buildVertices() {
        let tan80 = Math.tan(a2r(85));
        let sin80 = Math.sin(a2r(85));
        const bottomWidth = (ladleParams.width - (ladleParams.height))
        this.tudishLeftVertices = new Float32Array(
            [
                -ladleParams.width, (ladleParams.height * tan80),
                -(ladleParams.width - ((ladleParams.borderHeight / sin80))), (ladleParams.height * tan80),
                -(ladleParams.width - (ladleParams.height)), 0,
                (-(ladleParams.width - ((ladleParams.borderHeight / sin80)) - (ladleParams.height * tan80 - ladleParams.borderHeight) / tan80)), ladleParams.borderHeight,
                -(bottomWidth - ladleParams.gap / 2), 0,
                -(bottomWidth - ladleParams.gap / 2), ladleParams.borderHeight
            ]
        );
        let toReverseTudishLeftVertices = new Float32Array(12);

        toReverseTudishLeftVertices[8] = (bottomWidth / 2 - ladleParams.gap / 2)
        toReverseTudishLeftVertices[10] = (bottomWidth / 2 - ladleParams.gap / 2)
        this.tudishRightVertices = verticeUtils.reverse(this.tudishLeftVertices);



        this.tudishLeftInsideVertices = new Float32Array(
            [
                (-(ladleParams.width) + animConfig.borderThinkness / sin80), ((ladleParams.height * tan80) - animConfig.borderThinkness),
                (-(ladleParams.width - ((ladleParams.borderHeight / sin80))) - animConfig.borderThinkness / sin80), ((ladleParams.height * tan80) - animConfig.borderThinkness),
                (-(ladleParams.width - (ladleParams.height)) + animConfig.borderThinkness / sin80), (0 + animConfig.borderThinkness),
                ((-(ladleParams.width - ((ladleParams.borderHeight / sin80)) - (ladleParams.height * tan80 - ladleParams.borderHeight) / tan80)) - animConfig.borderThinkness), (ladleParams.borderHeight - animConfig.borderThinkness),
                (-(bottomWidth - ladleParams.gap) - animConfig.borderThinkness), (0 + animConfig.borderThinkness),
                (-(bottomWidth - ladleParams.gap) - animConfig.borderThinkness), (ladleParams.borderHeight - animConfig.borderThinkness)
            ]
        );

        this.tudishRightInsideVertices = verticeUtils.reverse(this.tudishLeftInsideVertices);
    },
    buildAnimObj(gl) {
        this.leftTudish = AnimObjHelper.AnimObj(this.tudishLeftVertices, ladleParams.color, gl.TRIANGLE_STRIP, 2);
        this.leftTudishInside = AnimObjHelper.AnimObj(this.tudishLeftInsideVertices, new Float32Array([0.8, 0.8, 0.8, 1]), gl.TRIANGLE_STRIP, 2);
        this.rightTudish = AnimObjHelper.AnimObj(this.tudishRightVertices, ladleParams.color, gl.TRIANGLE_STRIP, 2);
        this.rightTudishInside = AnimObjHelper.AnimObj(this.tudishRightInsideVertices, new Float32Array([0.8, 0.8, 0.8, 1]), gl.TRIANGLE_STRIP, 2)
    },
    buildBundle() {
        this.bundle = AnimObjHelper.AnimObjBundle([this.leftTudish, this.leftTudishInside, this.rightTudish, this.rightTudishInside], [0, 59 + 2000, 0])
    },
    build(gl) {
        this.buildVertices();
        this.buildAnimObj(gl);
        this.buildBundle();
        return this.bundle;
    }
}