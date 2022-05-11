SELECT click_id, call_time, call_comment, id
FROM online.lead_call_log lcl 
WHERE click_id in (
		SELECT DISTINCT click_id
		FROM online.lead_data ld 
		WHERE updated_at >= '2022-01-01 00:00:00');		
