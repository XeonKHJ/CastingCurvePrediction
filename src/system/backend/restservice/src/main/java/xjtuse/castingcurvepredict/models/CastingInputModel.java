package xjtuse.castingcurvepredict.models;

import java.util.Dictionary;
import java.util.Hashtable;

public class CastingInputModel {
    private int param = 0;
    private Dictionary<String, Object> _keyValues = new Hashtable<String, Object>();
    public CastingInputModel()
    {
        
    }

    public Dictionary<String, Object> getKeyValues()
    {
        return _keyValues;
    }

    public int getParam()
    {
        return param;
    }
}
