package xjtuse.castingcurvepredict.castingpredictiors;

import java.util.HashMap;

public class GeneratorInput {
    private int param = 0;
    private HashMap<String, Object> _keyValues = new HashMap<String, Object>();

    public GeneratorInput() {

    }

    public HashMap<String, Object> getKeyValues() {
        return _keyValues;
    }

    public int getParam() {
        return param;
    }
}
