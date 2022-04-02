package xjtuse.castingcurvepredict.models;

import java.util.Dictionary;
import java.util.HashMap;

public class CastingInputModel {
    private int param = 0;
    private HashMap<String, Object> _keyValues = new HashMap<String, Object>();

    public CastingInputModel() {

    }

    public HashMap<String, Object> getKeyValues() {
        return _keyValues;
    }

    public int getParam() {
        return param;
    }
}
