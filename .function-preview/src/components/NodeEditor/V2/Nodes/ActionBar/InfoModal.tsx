import { OperationVersion } from "@cognite/calculation-backend";
import { Modal } from "@cognite/cogs.js";
import { useEffect, useState } from "react";
import styled from "styled-components";
import Markdown from "../../../../Markdown/Markdown";

type Props = {
  indslFunction: OperationVersion;
  isOpen?: boolean;
  onClose?: () => void;
};

const InfoModal = ({
  indslFunction,
  isOpen = false,
  onClose = () => {},
}: Props) => {
  const [isModalVisible, setIsModalVisible] = useState(isOpen);

  // Control Uncontrolled/Controlled State
  useEffect(() => {
    setIsModalVisible(isOpen);
  }, [isOpen]);

  return (
    <ModalWrapper
      appElement={document.getElementsByTagName("body")}
      title={indslFunction.name}
      visible={isModalVisible}
      footer={null}
      onCancel={() => {
        setIsModalVisible(false);
        onClose();
      }}
      width={750}
    >
      <Markdown>{indslFunction.description || ""}</Markdown>
    </ModalWrapper>
  );
};

export default InfoModal;

const ModalWrapper = styled(Modal)`
  .cogs-modal-header {
    border-bottom: none;
    font-size: var(--cogs-t3-font-size);
  }
`;
