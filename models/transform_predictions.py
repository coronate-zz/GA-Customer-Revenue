


def transform(pred_test):
	sub_df = pd.DataFrame({"fullVisitorId":test_id})
	pred_test[pred_test<0] = 0
	sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
	sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
	sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
	sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
	return sub_df


