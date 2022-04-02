package xjtuse.castingcurvepredict.viewmodels;

import java.util.ArrayList;
import java.util.List;

public class ModelCollectionViewModel {
    private List<MLModelViewModel> mlModelViewModels = new ArrayList<MLModelViewModel>();

    public ModelCollectionViewModel(List<MLModelViewModel> viewModels)
    {
        mlModelViewModels = viewModels;
    }

    public List<MLModelViewModel> getMlModelViewModels()
    {
        return mlModelViewModels;
    }
}
