package xjtuse.castingcurvepredict.data;

public class Project {
    private long _id;
    private String _machineName;
    private Double _tudishWidth;

    public void setId(long id)
    {
        _id = id;
    }

    public long getId()
    {
        return _id;
    }

    public void setMachineName(String machineName)
    {
        _machineName = machineName;
    }

    public String getMachineName()
    {
        return _machineName;
    }

    public void setTudishWidth(Double tudishWidth)
    {
        _tudishWidth = tudishWidth;
    }

    public Double getTudishWidth()
    {
        return _tudishWidth;
    }
}
