SELECT click_id, 
		any(age) as age, any(gender) as gender, 
		MAX(fake_approve) as fake_approve, 
		MAX(call_time) - MIN(call_time) as var, 
		anyHeavy(comment) as comment,
		MAX(IF(call_comment == 'Продажа', 1, 0)) as isSuccsess,
		MAX(IF(call_comment == 'Продажа', call_time, CAST('1970-01-01 00:00:00' as DateTime))) as saleTimestamp,
		groupUniqArray(call_time) as callTimeSeries,
		groupUniqArray(call_comment) as callCommentArr
FROM
(SELECT click_id, call_time, call_comment, id
FROM online.lead_call_log lcl 
WHERE click_id in (
		SELECT DISTINCT click_id
		FROM online.lead_data ld 
		WHERE updated_at >= '2022-01-01 00:00:00')) b
INNER JOIN (
	SELECT id, click_id, lead_id, data, comment, age, gender, fake_approve
	FROM online.lead_data
	WHERE updated_at >= '2022-01-01 00:00:00') a ON a.click_id == b.click_id
GROUP BY click_id;		
