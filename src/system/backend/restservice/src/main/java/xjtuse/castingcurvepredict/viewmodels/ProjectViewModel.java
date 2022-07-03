package xjtuse.castingcurvepredict.viewmodels;

import xjtuse.castingcurvepredict.data.Project;

public class ProjectViewModel {
    private long _id;
    private String _machineName;
    private Double _tudishWidth;

    public ProjectViewModel(Project dataModel)
    {
        _id = dataModel.getId();
        _machineName = dataModel.getMachineName();
        _tudishWidth = dataModel.getTudishWidth();
    }

    public long getId()
    {
        return _id;
    }

    public String getMachineName()
    {
        return _machineName;
    }

    public Double getTudishWidth()
    {
        return _tudishWidth;
    }
}
