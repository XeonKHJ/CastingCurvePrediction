package xjtuse.castingcurvepredict.castingpredictiors;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import xjtuse.castingcurvepredict.interfaces.*;
import xjtuse.castingcurvepredict.models.*;

public class JsonFileCastingGenerator implements ICastingGenerator {

    @Override
    public CastingResultModel PredcitCastingCurve(CastingInputModel input) {
        // TODO Auto-generated method stub
        File file = (File) input.getKeyValues().get("file");
        CastingResultModel resultModel = new CastingResultModel();
        BufferedReader bfReader = null;
        if (file != null && file.canRead()) {
            try {
                bfReader = new BufferedReader(new FileReader(file));
                Boolean firstLine = true;
                String line;
                while ((line = bfReader.readLine()) != null) {
                    if (firstLine) {
                        firstLine = false;
                    } else {
                        String itemToSplit = line;
                        String[] item = itemToSplit.split(",");
                        String date = item[0];
                        double value = Double.parseDouble(item[1]);
                        resultModel.addResultItem(date, value);
                    }
                }
            } catch (NumberFormatException | IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }

        return resultModel;
    }

    @Override
    public ArrayList<PredictionModel> getModelList() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void updateModel(CastingModel data) {
        // TODO Auto-generated method stub

    }

    @Override
    public void updateModel(ArrayList<CastingModel> datas) {
        // TODO Auto-generated method stub

    }

}
