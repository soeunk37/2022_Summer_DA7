-- 해당 쿼리에서 사용하는 RFM 테이블은 DEMO 테이블, 상품 구매 정보 테이블, 상품 분류 정보 테이블을 결합 및 단순 조작 (년/월/일 구분 등) 하고 유저별 P_등급 (전체 결제 RFM 등급) 을 조합한 테이블임.
-- 참고 (구매테이블_RFM_df)

-- 최종 테이블셋 구성을 위해 RFM 테이블에 유저별 LPAY 결제 건수, 금액 추가 
With check_ AS (

SELECT 고객번호 
        , 장바구니_식별번호
        , 온오프_구분
        , 제휴사_구분
        , 상품_구분
        , 구매일자
        , 구매시간
        , 구매금액
        , 구매수량
        , 상품_소분류
        , 상품_중분류
        , 성별
        , 연령
        , 거주지_대분류
        , 년
        , 월
        , 일
        , 월_주문
        ,'유통사' AS 구매타입 
        , 등급 
FROM rmf r
)
, lpay_add AS (
SELECT 고객번호
        , 장바구니_식별번호
        , 온오프_구분
        , 제휴사_구분
        , 상품_구분
        , 구매일자
        , 구매시간
        , 구매금액
        , 구매수량
        , 상품_소분류
        , 상품_중분류
        , 성별
        , 연령
        , 거주지_대분류
        , 년
        , 월
        , 일
        , 월_주문
        , 구매타입
        , lpay_cnt AS lpay_결제횟수 
        , lpay_sum AS lpay_결제액수
        , t.등급 
FROM check_ t
LEFT JOIN 
              (SELECT cust, count(rct_no) AS lpay_cnt, sum(buy_am) AS lpay_sum FROM lpay GROUP BY cust) l
ON t.고객번호 = l.cust
) 

-- p등급, l등급 수정

,fin AS (
SELECT c.*, cd.p_등급, td.l_등급
FROM lpay_add c
LEFT JOIN (SELECT distinct 고객번호, 등급 AS p_등급  FROM rmf) cd
ON c.고객번호  = cd.고객번호 
LEFT JOIN (SELECT distinct 고객번호, 등급 AS l_등급  FROM lpay_rfm) td
ON c.고객번호 = td.고객번호  )

-- 사용 데이터 셋  ( 해당 데이터 0726final 테이블로 export)
SELECT 고객번호, 장바구니_식별번호, 온오프_구분, 제휴사_구분, 상품_구분, 구매일자, 구매시간, 구매금액, 구매수량, 상품_소분류, 상품_중분류, 성별, 연령, 거주지_대분류, 년,월, 일, 월_주문, 구매타입, lpay_결제횟수,lpay_결제액수, p_등급, l_등급 
FROM fin
WHERE 구매타입 = '유통사';




