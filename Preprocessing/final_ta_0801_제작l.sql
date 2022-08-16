-- 파생변수를 포함한 최종 테이블셋 구성을 위해 결제액수 평균을 결제총액 / 구매수량 이 아닌 결제총액 / 장바구니 번호 갯수 로 수정 
-- final _0730 = fin_df_0730 데이터 

WITH base_ AS (
SELECT * 
FROM final_0730
LEFT JOIN (SELECT 고객번호, count(장바구니_식별번호) AS P_결제횟수, sum(구매금액) AS P_결제금액, ROUND( sum(구매금액) / count(장바구니_식별번호), 4) AS P_결제액수평균_1
           FROM final_ta -- 사용 테이블셋 () 
           GROUP BY 고객번호)
USING (고객번호)
)

SELECT 고객번호
    , 온오프_구분
    , 제휴사_구분
    , 구매일자
    , 구매시간
    , 구매금액_X
    , 성별
    , 연령
    , 거주지_대분류
    , 년
    , 월 
    , 일
    , 구매타입
    , LPAY_결제횟수
    , LPAY_결제액수
    , P_등급
    , L_등급
    , 최근구매일_R
    , 상품중분류_정규화
    , 주중주말
    , 구매금액_Y AS 요일별_인당_평균결제금액
    , P_결제금액
    , P_결제횟수
    , P_결제액수평균_1 AS P_결제액수평균 
FROM base_;

-- 해당 데이터셋 final_ta_0801 테이블 로 export


