import styled from "styled-components/macro";

export const SourceCircle = styled.span`
  height: 20px;
  width: 20px;
  background-color: ${(props) => props.color || "lightgreen"};
  margin: 20px;
  margin-left: 5px;
  border-radius: 50%;
  flex-shrink: 0;
  opacity: ${(props: { fade?: boolean }) => (props.fade ? "0.2" : "1")};
`;

export const SourceSquare = styled.span`
  height: 20px;
  width: 20px;
  background-color: ${(props) => props.color || "lightgreen"};
  margin: 20px;
  margin-left: 5px;
  flex-shrink: 0;
  opacity: ${(props: { fade?: boolean }) => (props.fade ? "0.2" : "1")};
`;
