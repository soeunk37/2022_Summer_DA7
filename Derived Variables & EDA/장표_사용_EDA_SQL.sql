-- DEMO : 고객 데모 정보 테이블 
-- PDDE : 유통사 상품 구매 내역 테이블 
-- COP_U : 제휴사 이용 정보 테이블 
-- PD_CLAC : 상품 분류 정보 테이블 
-- LPAY : 엘페이 이용 테이블 

-- [1] LPAY 결제 이력 유저 비율 확인 결제이력 X : 결제이력 O (70:30)
SELECT   CASE WHEN l.cust is null THEN '결제이력_X' ELSE '결제이력_O' END AS LPAY
       , count(distinct t.cust)
FROM (
        SELECT distinct cust FROM pdde
        
        UNION 
        
        SELECT distinct cust FROM cop_u
) T
LEFT JOIN (
                SELECT cust 
                FROM lpay
            ) L
ON       t.cust = l.cust
GROUP BY CASE WHEN l.cust is null THEN '결제이력_X'ELSE '결제이력_O' END;

-- [2] 전체 결제 대비 LPAY 결제 현황 
    -- [2]-1 전체 결제 금액 / 유저 수


SELECT  chnl_dv, sum(buy_am) AS total_amt, count(distinct cust) AS total_cnt
FROM (
        SELECT cust, buy_am, chnl_dv FROM pdde
        
        UNION ALL
        
        SELECT cust, buy_am, chnl_dv FROM cop_u
) T
GROUP BY chnl_dv

    -- [2]-2 LPAY 결제 금액 / 유저 수

SELECT  chnl_dv, sum(buy_am) AS total_amt, count(distinct cust) AS total_cnt
FROM lpay
GROUP BY chnl_dv;

    -- [2]-3 전체 비중 결과 및 시각화는 엑셀로 진행
    
    
-- [3]-1 LPAY 결제 이력 유무 별 월 ARPU (온라인)

SELECT 
         CASE WHEN coalesce(lpay_cnt,0) > 0 THEN '이력있음' ELSE '이력없음' END AS l
        , count(distinct t.cust)
        , sum(buy_am)
        , sum(buy_am)/count(distinct t.cust)/12
FROM (
        SELECT cust, buy_am 
        FROM pdde 
        WHERE chnl_dv = 2
        
        UNION ALL
        
        SELECT cust, buy_am 
        FROM cop_u 
        WHERE chnl_dv = 2
    ) t
LEFT JOIN (
                SELECT cust, count(rct_no) AS lpay_cnt
                FROM lpay
                GROUP BY cust
            ) l
ON       t.cust = l.cust
GROUP BY CASE WHEN coalesce(lpay_cnt,0) > 0 THEN '이력있음' ELSE '이력없음' END;

-- [4] P_등급별 제휴사/유통사 수
-- base_ : 

SELECT l_등급, 고객번호, count(distinct 제휴사_구분) AS 제휴사_수 
FROM base_ 
GROUP BY l_등급, 고객번호 ;

-- 아래 작업 이후 출력 값 별 엑셀 시각화 및 CLASS 후 비율 처리 작업 진행.

-- [5] LPAY 등급별 결제총액 CLASS 비율

SELECT 고객번호,l_등급, SUM(구매금액_x)
FROM base_ 
WHERE l_등급 is not null 
    AND 온오프_구분 = 2
GROUP BY 고객번호,l_등급
ORDER BY 고객번호,l_등급;


-- [5] LPAY 등급별 결제액 평균 CLASS 비율
-- base_ = final_ta_0801.csv 

SELECT 고객번호,l_등급, AVG(구매금액_x)
FROM base_ 
WHERE l_등급 is not null 
    AND 온오프_구분 = 2
GROUP BY 고객번호,l_등급
ORDER BY 고객번호,l_등급;

--[6] LPAY 등급별 결제횟수 CLASS 비율

SELECT 고객번호,l_등급, count(*)
FROM base_ 
WHERE l_등급 is not null 
    AND 온오프_구분 = 2
GROUP BY 고객번호,l_등급
ORDER BY 고객번호,l_등급;

--[7] LPAY 등급별 연령 비중 

SELECT l_등급, 연령, count(고객번호) 
FROM base_ 
WHERE l_등급 is not null 
    AND 온오프_구분 = 2
GROUP BY l_등급, 연령
ORDER BY l_등급, 연령

--[8] LPAY 등급별 P_등급 

SELECT l_등급,p_등급, count(distinct 고객번호) 
FROM base_ 
WHERE l_등급 is not null 
    AND l_등급 != 0 
    AND 온오프_구분 = 2
GROUP BY l_등급,p_등급
ORDER BY l_등급,p_등급;