package xjtuse.castingcurvepredict.viewmodels;

import java.util.ArrayList;
import java.util.List;

public class ProjectCollectionViewModel {
    private List<ProjectViewModel> mlModelViewModels = new ArrayList<ProjectViewModel>();

    public ProjectCollectionViewModel(List<ProjectViewModel> viewModels)
    {
        mlModelViewModels = viewModels;
    }

    public List<ProjectViewModel> getProjectViewModels()
    {
        return mlModelViewModels;
    }
}
