비즈니스에 적용될 만한 내용들에 대해 인공지능과 기계학습을 적용했습니다.

Decision maker
- 구매 요청 처리 과정에서 1)기술 담당자에게 결재안을 검토 요청 해야할까??, 2)아니면 재정 담당자에게 바로 보낼까?? 라는 의사결정에 대한 판단을 기계학습에게 맡기는 케이스입니다.
- XG Booster 모델을 사용했으며, Target은 기술 담당자에게 검토 요청에 대한 tech_approval_required 컬럼입니다.
- Input은 Categorical data(requester_id, role, product), Numeric data(quantity, price, total)로 구성됩니다.
- 10% 이상의 correlation 데이터만 사용했으며, categorical data는 pandas.get_dummies를 이용해 임베딩했습니다.
